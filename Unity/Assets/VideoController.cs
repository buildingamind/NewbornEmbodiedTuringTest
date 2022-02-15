using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public class VideoController : MonoBehaviour
{
    VideoPlayer videoPlayer;
    float remainingTime;

    void Awake()
    {
        videoPlayer = gameObject.GetComponent<VideoPlayer>();
        videoPlayer.playOnAwake = false;
        videoPlayer.skipOnDrop = true;
        videoPlayer.isLooping = true;
        videoPlayer.sendFrameReadyEvents = true;

        videoPlayer.prepareCompleted += VideoStarted;

        videoPlayer.Pause();
        videoPlayer.Stop();
    }

    private void VideoStarted(VideoPlayer vp)
    {
        vp.Pause();
        remainingTime = 0.0f;
    }

    // Set video URL as the source of the video player.
    public void SetVideoSource(string videoUrl)
    {
        // Stop the video player if source is not specified.
        if (String.IsNullOrEmpty(videoUrl))
        {
            videoPlayer.clip = null;
            videoPlayer.source = VideoSource.VideoClip;
            videoPlayer.Stop();
        }
        else
        {
            videoPlayer.url = videoUrl;
            videoPlayer.source = VideoSource.Url;
            videoPlayer.Prepare();
        }
    }

    private void UpdateVideoFrames()
    {
        // Manually update the video frames in order to get playback speed
        // consistent with the Time.timeScale.

        if (videoPlayer.isPrepared && videoPlayer.frame >= 0)
        {
            // Calculate elapsed time since last frame update.
            float elapsedTime = Time.deltaTime + remainingTime;

            // Calculate maximum number of frames to step without going
            // over the elapsed time.
            int framesToStep = (int) (videoPlayer.frameRate * elapsedTime) ;

            // Remember the time remaining after stepping forward.
            remainingTime = elapsedTime - (float) framesToStep / videoPlayer.frameRate;

            // Step forward with the number of frames.
            for (int i = 0; i < framesToStep; i++)
            {
                videoPlayer.StepForward();

                // Restart the video if the end is reached.
                if (videoPlayer.frame == (long) videoPlayer.frameCount - 1)
                {
                    videoPlayer.Stop();
                    videoPlayer.Prepare();
                }
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        UpdateVideoFrames();
    }
}
