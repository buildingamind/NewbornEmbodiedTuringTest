using System;
using System.IO;

public class SimpleLogger
{
    StreamWriter logWriter;

    /// Create directories and log file in the specified path.
    public SimpleLogger(string logPath)
    {
        var logDir = Path.GetDirectoryName(logPath);
        if (!String.IsNullOrEmpty(logDir))
            Directory.CreateDirectory(logDir);

        logWriter = new StreamWriter(logPath, true);
    }

    /// Record a single line in the log file.
    public void Log(string str)
    {
        logWriter.WriteLine(str);
        logWriter.Flush();
    }

    public void Log(string format, params object[] args)
    {
        string str = String.Format(format, args);
        Log(str);
    }
}