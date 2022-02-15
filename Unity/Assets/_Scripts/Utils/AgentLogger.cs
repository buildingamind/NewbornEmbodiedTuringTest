using System;
using System.IO;

public class AgentLogger : SimpleLogger
{
    string agentName;

    public AgentLogger(string logDir, string agentName) :
        base(Path.Combine(logDir, agentName + ".txt"))
    {
        this.agentName = agentName;
        LogHeader();
    }

    public void LogHeader()
    {
        Log("Episode, Step, {0}.x, {0}.y, {0}.angle", agentName);
    }

    public void LogStep(int episode, int step, float x, float z, float angle)
    {
        Log("{0}, {1}, {2}, {3}, {4}", episode, step, x, z, angle);
    }
}
