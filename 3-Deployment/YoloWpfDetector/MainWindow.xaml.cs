using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace YoloWpfDetector;

public partial class MainWindow : Window
{
    private Process? _detectorProcess;
    private CancellationTokenSource? _streamCancellation;
    private Task? _detectorLoopTask;
    private bool _isStopping;
    private readonly List<ModelOption> _models;
    private readonly string _workerPath;

    public MainWindow()
    {
        InitializeComponent();

        _workerPath = Path.Combine(AppContext.BaseDirectory, "detector_worker.py");
        _models = FindAvailableModels();
        ModelComboBox.Items.Add(new ModelOption("No model", null));
        foreach (var model in _models)
        {
            ModelComboBox.Items.Add(model);
        }

        ModelComboBox.SelectedIndex = 0;
        UpdateSelectedModelPath();
        ConfidenceValueTextBlock.Text = FormatConfidence(GetSelectedConfidence());

        _ = LoadModelClassesAsync();
    }

    private void StartButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_detectorProcess is not null)
            {
                return;
            }

            if (!File.Exists(_workerPath))
            {
                SetStatus($"Detector worker was not found: {_workerPath}");
                return;
            }

            var modelPath = GetSelectedModelPath();
            if (modelPath is not null && !File.Exists(modelPath))
            {
                SetStatus($"Selected model was not found: {modelPath}");
                return;
            }

            var sourceMode = SourceComboBox.SelectedIndex == 0 ? "camera" : "video";
            var sourceValue = sourceMode == "camera" ? CameraIndexTextBox.Text.Trim() : VideoPathTextBox.Text.Trim();

            if (sourceMode == "camera" && !int.TryParse(sourceValue, out _))
            {
                SetStatus("Camera index must be a number. Use 0 for the default local camera.");
                return;
            }

            if (sourceMode == "video" && !File.Exists(sourceValue))
            {
                SetStatus("Choose a valid MP4 video before starting.");
                return;
            }

            var fpsLimit = GetSelectedFpsLimit();
            _streamCancellation = new CancellationTokenSource();

            var startInfo = new ProcessStartInfo
            {
                FileName = "python",
                WorkingDirectory = AppContext.BaseDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            startInfo.ArgumentList.Add(_workerPath);
            if (modelPath is not null)
            {
                startInfo.ArgumentList.Add("--model");
                startInfo.ArgumentList.Add(modelPath);
            }

            startInfo.ArgumentList.Add("--source-mode");
            startInfo.ArgumentList.Add(sourceMode);
            startInfo.ArgumentList.Add("--source");
            startInfo.ArgumentList.Add(sourceValue);
            startInfo.ArgumentList.Add("--fps");
            startInfo.ArgumentList.Add(fpsLimit.ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--confidence");
            startInfo.ArgumentList.Add(GetSelectedConfidence().ToString(CultureInfo.InvariantCulture));

            _detectorProcess = Process.Start(startInfo);
            if (_detectorProcess is null)
            {
                SetStatus("Could not start the detector process.");
                return;
            }

            SetRunningState(true);
            var runMode = modelPath is null ? "preview" : "detection";
            SetStatus($"Starting {sourceMode} {runMode} at {FormatFpsLimit(fpsLimit)}...");

            _detectorLoopTask = Task.Run(() => RunDetectorLoopAsync(_detectorProcess, _streamCancellation.Token));
        }
        catch (Exception ex)
        {
            SetStatus($"Detector error: {ex.Message}");
            _ = StopDetectorAsync(waitForLoop: false);
        }
    }

    private async void StopButton_Click(object sender, RoutedEventArgs e)
    {
        await StopDetectorAsync();
        SetStatus("Detection stopped.");
    }

    private void BrowseButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Choose MP4 video",
            Filter = "MP4 video (*.mp4)|*.mp4|Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All files (*.*)|*.*"
        };

        if (dialog.ShowDialog(this) == true)
        {
            VideoPathTextBox.Text = dialog.FileName;
        }
    }

    private void SourceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsInitialized)
        {
            return;
        }

        var isCamera = SourceComboBox.SelectedIndex == 0;
        CameraIndexTextBox.IsEnabled = isCamera;
        VideoPathTextBox.IsEnabled = !isCamera;
        BrowseButton.IsEnabled = !isCamera;
        PlaceholderTextBlock.Text = isCamera ? "Press Start to open the local camera." : "Choose an MP4, then press Start.";
        SetStatus(isCamera ? "Ready. Local camera is selected." : "Ready. MP4 video source is selected.");
    }

    private void ModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsInitialized)
        {
            return;
        }

        UpdateSelectedModelPath();
        SetStatus($"Selected model: {GetSelectedModelName()}");
        _ = LoadModelClassesAsync();
    }

    private void ConfidenceSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (ConfidenceValueTextBlock is not null)
        {
            ConfidenceValueTextBlock.Text = FormatConfidence(e.NewValue);
        }
    }

    private async void RequirementsButton_Click(object sender, RoutedEventArgs e)
    {
        RequirementsButton.IsEnabled = false;
        SetStatus("Checking Python requirements...");

        try
        {
            var installed = await AreRequirementsInstalledAsync();
            if (installed)
            {
                SetStatus("Requirements are already installed.");
                return;
            }

            SetStatus("Missing requirements found. Installing from requirements.txt...");
            var result = await RunPythonProcessAsync(["-m", "pip", "install", "-r", "requirements.txt"], TimeSpan.FromMinutes(10));
            if (result.ExitCode == 0)
            {
                SetStatus("Requirements installed successfully.");
            }
            else
            {
                SetStatus($"Requirements install failed: {TrimProcessMessage(result.Error, result.Output)}");
            }
        }
        catch (Exception ex)
        {
            SetStatus($"Requirements check failed: {ex.Message}");
        }
        finally
        {
            RequirementsButton.IsEnabled = _detectorProcess is null;
        }
    }

    private async void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        await StopDetectorAsync();
    }

    private async Task RunDetectorLoopAsync(Process process, CancellationToken cancellationToken)
    {
        try
        {
            var statusTask = ReadStatusAsync(process, cancellationToken);
            await ReadFramesAsync(process, cancellationToken);
            await statusTask;
        }
        catch (OperationCanceledException)
        {
            await Dispatcher.InvokeAsync(() => SetStatus("Detection stopped."));
        }
        catch (Exception ex)
        {
            await Dispatcher.InvokeAsync(() => SetStatus($"Detector error: {ex.Message}"));
        }
        finally
        {
            await StopDetectorAsync(waitForLoop: false);
        }
    }

    private async Task ReadFramesAsync(Process process, CancellationToken cancellationToken)
    {
        var stream = process.StandardOutput.BaseStream;
        var lengthBuffer = new byte[4];

        while (!cancellationToken.IsCancellationRequested && !process.HasExited)
        {
            if (!await ReadExactAsync(stream, lengthBuffer, cancellationToken))
            {
                break;
            }

            var length = BitConverter.ToInt32(lengthBuffer, 0);
            if (length <= 0 || length > 25_000_000)
            {
                throw new InvalidDataException("The detector stream sent an invalid frame.");
            }

            var frameBuffer = new byte[length];
            if (!await ReadExactAsync(stream, frameBuffer, cancellationToken))
            {
                break;
            }

            await Dispatcher.InvokeAsync(() =>
            {
                VideoImage.Source = CreateBitmap(frameBuffer);
                PlaceholderTextBlock.Visibility = Visibility.Collapsed;
            });
        }
    }

    private async Task ReadStatusAsync(Process process, CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested && !process.HasExited)
        {
            var line = await process.StandardError.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                break;
            }

            if (!string.IsNullOrWhiteSpace(line))
            {
                await Dispatcher.InvokeAsync(() => SetStatus(line));
            }
        }
    }

    private async Task StopDetectorAsync(bool waitForLoop = true)
    {
        if (_isStopping)
        {
            return;
        }

        _isStopping = true;
        _streamCancellation?.Cancel();

        if (_detectorProcess is not null)
        {
            try
            {
                if (!_detectorProcess.HasExited)
                {
                    _detectorProcess.Kill(entireProcessTree: true);
                    await _detectorProcess.WaitForExitAsync();
                }
            }
            catch
            {
                // The process may already be gone while the UI is closing.
            }
            finally
            {
                _detectorProcess.Dispose();
                _detectorProcess = null;
            }
        }

        if (waitForLoop && _detectorLoopTask is not null)
        {
            try
            {
                await _detectorLoopTask;
            }
            catch
            {
                // The loop reports user-facing errors through the status area.
            }
        }

        _detectorLoopTask = null;
        _streamCancellation?.Dispose();
        _streamCancellation = null;
        _isStopping = false;
        if (Dispatcher.CheckAccess())
        {
            SetRunningState(false);
        }
        else
        {
            await Dispatcher.InvokeAsync(() => SetRunningState(false));
        }
    }

    private void SetRunningState(bool isRunning)
    {
        StartButton.IsEnabled = !isRunning;
        StopButton.IsEnabled = isRunning;
        SourceComboBox.IsEnabled = !isRunning;
        CameraIndexTextBox.IsEnabled = !isRunning && SourceComboBox.SelectedIndex == 0;
        VideoPathTextBox.IsEnabled = !isRunning && SourceComboBox.SelectedIndex == 1;
        BrowseButton.IsEnabled = !isRunning && SourceComboBox.SelectedIndex == 1;
        FpsComboBox.IsEnabled = !isRunning;
        ConfidenceSlider.IsEnabled = !isRunning;
        ModelComboBox.IsEnabled = !isRunning;
        RequirementsButton.IsEnabled = !isRunning;
    }

    private void SetStatus(string message)
    {
        StatusTextBlock.Text = message;
    }

    private double GetSelectedFpsLimit()
    {
        if (FpsComboBox.SelectedItem is ComboBoxItem item &&
            item.Tag is not null &&
            double.TryParse(item.Tag.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var taggedFps))
        {
            return taggedFps;
        }

        if (FpsComboBox.SelectedItem is ComboBoxItem itemByContent &&
            double.TryParse(itemByContent.Content?.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var fps))
        {
            return fps;
        }

        return 0;
    }

    private static string FormatFpsLimit(double fpsLimit)
    {
        return fpsLimit <= 0 ? "maximum available FPS" : $"{fpsLimit:0} FPS";
    }

    private double GetSelectedConfidence()
    {
        return Math.Clamp(ConfidenceSlider.Value, 0.05, 0.95);
    }

    private static string FormatConfidence(double confidence)
    {
        return $"{confidence:P0}";
    }

    private async Task LoadModelClassesAsync()
    {
        try
        {
            var modelPath = GetSelectedModelPath();
            if (modelPath is null)
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    ModelClassesListBox.Items.Clear();
                    ModelClassesListBox.Items.Add("No model selected.");
                    ClassesHeaderTextBlock.Text = "Classes";
                });
                return;
            }

            await Dispatcher.InvokeAsync(() =>
            {
                ModelClassesListBox.Items.Clear();
                ModelClassesListBox.Items.Add("Loading model classes...");
                ClassesHeaderTextBlock.Text = "Classes";
            });

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(45));
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    WorkingDirectory = AppContext.BaseDirectory,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            process.StartInfo.ArgumentList.Add(_workerPath);
            process.StartInfo.ArgumentList.Add("--model");
            process.StartInfo.ArgumentList.Add(modelPath);
            process.StartInfo.ArgumentList.Add("--list-classes");

            process.Start();
            var outputTask = process.StandardOutput.ReadToEndAsync(timeout.Token);
            var errorTask = process.StandardError.ReadToEndAsync(timeout.Token);
            await process.WaitForExitAsync(timeout.Token);

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException(string.IsNullOrWhiteSpace(error) ? "Could not read model classes." : error.Trim());
            }

            var classes = JsonSerializer.Deserialize<List<string>>(output) ?? [];
            await Dispatcher.InvokeAsync(() =>
            {
                ModelClassesListBox.Items.Clear();
                foreach (var modelClass in classes)
                {
                    ModelClassesListBox.Items.Add(modelClass);
                }

                ClassesHeaderTextBlock.Text = $"Classes ({classes.Count})";
            });
        }
        catch (Exception ex)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                ModelClassesListBox.Items.Clear();
                ModelClassesListBox.Items.Add("Could not load classes.");
                ClassesHeaderTextBlock.Text = "Classes";
                SetStatus($"Class list error: {ex.Message}");
            });
        }
    }

    private async Task<bool> AreRequirementsInstalledAsync()
    {
        var result = await RunPythonProcessAsync(["-c", "import ultralytics, cv2"], TimeSpan.FromSeconds(30));
        return result.ExitCode == 0;
    }

    private async Task<ProcessResult> RunPythonProcessAsync(string[] arguments, TimeSpan timeout)
    {
        using var timeoutSource = new CancellationTokenSource(timeout);
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "python",
                WorkingDirectory = AppContext.BaseDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };

        foreach (var argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        var outputTask = process.StandardOutput.ReadToEndAsync(timeoutSource.Token);
        var errorTask = process.StandardError.ReadToEndAsync(timeoutSource.Token);
        await process.WaitForExitAsync(timeoutSource.Token);

        return new ProcessResult(
            process.ExitCode,
            await outputTask,
            await errorTask);
    }

    private static string TrimProcessMessage(string error, string output)
    {
        var message = string.IsNullOrWhiteSpace(error) ? output : error;
        message = message.Trim();
        return message.Length <= 220 ? message : $"{message[..220]}...";
    }

    private static BitmapImage CreateBitmap(byte[] jpegBytes)
    {
        using var memory = new MemoryStream(jpegBytes);
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = memory;
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    private static async Task<bool> ReadExactAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken)
    {
        var offset = 0;
        while (offset < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer.AsMemory(offset, buffer.Length - offset), cancellationToken);
            if (read == 0)
            {
                return false;
            }

            offset += read;
        }

        return true;
    }

    private string? GetSelectedModelPath()
    {
        if (ModelComboBox.SelectedItem is ModelOption model)
        {
            return model.Path;
        }

        return null;
    }

    private string GetSelectedModelName()
    {
        if (ModelComboBox.SelectedItem is ModelOption model)
        {
            return model.Name;
        }

        return "No model";
    }

    private void UpdateSelectedModelPath()
    {
        ModelPathTextBlock.Text = GetSelectedModelPath() ?? "Camera/video preview without detection.";
    }

    private static List<ModelOption> FindAvailableModels()
    {
        return FindModelFiles();
    }

    private static List<ModelOption> FindModelFiles()
    {
        var models = new List<ModelOption>();
        var modelDirectory = Path.Combine(AppContext.BaseDirectory, "models");

        if (!Directory.Exists(modelDirectory))
        {
            return models;
        }

        foreach (var path in Directory.EnumerateFiles(modelDirectory, "*.pt").OrderBy(Path.GetFileName))
        {
            models.Add(new ModelOption(Path.GetFileName(path), path));
        }

        return models;
    }

    private sealed record ModelOption(string Name, string? Path);
    private sealed record ProcessResult(int ExitCode, string Output, string Error);
}
