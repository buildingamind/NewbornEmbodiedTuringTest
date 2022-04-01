using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using Unity.MLAgents;

public class ChickAcademy : MonoBehaviour
{
    // Public attributes
    public String ImprintVideo;
    public String TestVideo;

    // Private attributes
    VideoPlayer rightMonitor;
    VideoPlayer leftMonitor;
    float videoStartTime;
    public VideoClip DefaultVideo;
    public ChickAgent agent;

    public void Awake()
    {

        Academy.Instance.OnEnvironmentReset += EnvironmentReset;

        // Find monitors in the scene.
        rightMonitor = GameObject.Find("RightMonitor").GetComponent<VideoPlayer>();
        leftMonitor = GameObject.Find("LeftMonitor").GetComponent<VideoPlayer>();

        // Register callbacks for VideoPlayer.prepareCompleted events.
        rightMonitor.prepareCompleted += VideoStarted;
        leftMonitor.prepareCompleted += VideoStarted;
        
        // Set monitors to loop when finished
        rightMonitor.isLooping = true;
        leftMonitor.isLooping = true;

        // Set video urls from command line arguments.
        if (!String.IsNullOrEmpty(ArgumentParser.Options.ImprintVideo))
        {
            ImprintVideo = ArgumentParser.Options.ImprintVideo;
        }
        if (!String.IsNullOrEmpty(ArgumentParser.Options.TestVideo))
        {
            TestVideo = ArgumentParser.Options.TestVideo;
        }

    }

    public void ResetMonitors(int episodeCount)
    {
        GameObject rightTarget = GameObject.FindWithTag("RightTarget");
        GameObject leftTarget = GameObject.FindWithTag("LeftTarget");

        if (episodeCount % 2 == 0)
        {
            SetVideoSource(rightMonitor, ImprintVideo);
            SetVideoSource(leftMonitor, TestVideo);
            SetLayerByName(leftTarget, "Default");
            if (!String.IsNullOrEmpty(ImprintVideo) || true)
            {
                SetLayerByName(rightTarget, "ImprintTarget");
                agent.target = rightTarget.transform;
            }
        }
        else
        {
            SetVideoSource(rightMonitor, TestVideo);
            SetVideoSource(leftMonitor, ImprintVideo);
            SetLayerByName(rightTarget, "Default");
            if (!String.IsNullOrEmpty(ImprintVideo) || true)
            {
                SetLayerByName(leftTarget, "ImprintTarget");
                agent.target = leftTarget.transform;
            }
        }
    }

    private void SetLayerByName(GameObject obj, string layerName)
    {
        if (obj != null)
        {
            obj.layer = LayerMask.NameToLayer(layerName);
        }
    }

    private void VideoStarted(VideoPlayer videoPlayer)
    {
        videoPlayer.Pause();
        videoStartTime = Time.time;
    }

    private void SetVideoSource(VideoPlayer videoPlayer, String videoUrl)
    {
        // Play default video clip if videoUrl is not specified.
        if (String.IsNullOrEmpty(videoUrl))
        {
            videoPlayer.clip = DefaultVideo;
            videoPlayer.source = VideoSource.VideoClip;
        }
        else
        {
            videoPlayer.url = videoUrl;
            videoPlayer.source = VideoSource.Url;
        }
    }

    public void Update()
    {
        UpdateVideoFrames(rightMonitor);
        UpdateVideoFrames(leftMonitor);
    }


    private void UpdateVideoFrames(VideoPlayer videoPlayer)
    {
        // Manually update the video frames in order to get playback speed
        // consistent with the Time.timeScale.
        // 1. Calculate (target frame number) = (frame rate) * (elapsed time).
        // 2. Step forward (n) frames to catch up with the (target frame number),
        //    where (n) = (target frame number) - (current frame number).
        if (videoPlayer.frame == -1)
        {
            videoPlayer.Prepare();
        }

        float elapsedTime = Time.time - videoStartTime;

        if (videoPlayer.isPrepared && videoPlayer.frame >= 0)
        {
            int targetFrame = (int) (videoPlayer.frameRate * elapsedTime);
            int framesToStep = Mathf.Max(targetFrame - (int) videoPlayer.frame, 0);

            for (int i = 0; i < framesToStep; i++)
            {
                videoPlayer.StepForward();
                double lastFrame = videoPlayer.time;
                if (videoPlayer.length - lastFrame < 1){
                    videoPlayer.Stop();
                    videoPlayer.Prepare();
                }
            }
        }
    }
    public void EnvironmentReset()
    {
    }

}
