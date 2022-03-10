using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mono.Options;

public static class ArgumentParser
{
    public class CommandLineOptions
    {
        public int InputResolution;
        public int EpisodeSteps;    // Number of steps per episode.
        public int Seed;
        public string LogDir;          // Directory to store log files.
        public string ImprintVideo;
        public string TestVideo;
        public bool TestMode;
        public bool EnableAgentInfoChannel;
        public bool RecordChamber;
        public bool RecordAgent;
    }

    private static CommandLineOptions options;
    public static CommandLineOptions Options
    {
        get
        {
            // Parse command line when this property is accessed for the first time.
            if (options == null) ParseCommandLineArgs();
            return options;
        }
    }

    private static void ParseCommandLineArgs()
    {
        var args = System.Environment.GetCommandLineArgs();
        var parser = new OptionSet() {
            {"input_resolution=", "size of visual input.",
                (int v) => options.InputResolution = v},

            {"episode_steps=", "number of steps per episode",
                (int v) => options.EpisodeSteps = v},
            
            {"env_seed=", "seed for the unity environment",
                (int v) => options.Seed = v},

            {"log_dir=", "directory to store log files",
                v => options.LogDir = v},

            {"imprint_video=", "URL of first video",
                v => options.ImprintVideo = v},

            {"test_video=", "URL of second video",
                v => options.TestVideo = v},

            {"record_chamber", "record the top down camera view",
                v => options.RecordChamber = v != null},

            {"record_agent", "record the first person view of agent",
                v => options.RecordAgent = v != null},

            {"test_mode", "run in test mode",
                v => options.TestMode = v != null},

            {"enable_agent-info-channel", "enable agentInfoChannel in ChickAgent.",
                v => options.EnableAgentInfoChannel = v != null}
        };

        options = new CommandLineOptions();
        try
        {
            parser.Parse(args);
        }
        catch (OptionException e)
        {
            Debug.Log("OptionException: " + e.ToString());
        }
    }
}
