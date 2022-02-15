using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CameraManager : MonoBehaviour
{
    public int framesPerSecond = 5;

    List<Camera> _cameras;
    Camera agentRecorder;
    Camera chamberRecorder;
    int _currentCamera;
    string logDir, agentDir, chamberDir;
    bool recordAgent;
    bool recordChamber;

    private float interval;
    private float timer = 0;
    private int frame_counter = 0;

    // Start is called before the first frame update.
    void Start()
    {
        _cameras = new List<Camera>();
        _currentCamera = 0;

        // Find all cameras in the scene.
        foreach (Camera c in Camera.allCameras)
        {
            if (c.CompareTag("AgentRecorder"))
            {
                agentRecorder = c;
            }
            else if (c.CompareTag("ChamberRecorder"))
            {
                chamberRecorder = c;
            }
            else
            {
                _cameras.Add(c);
            }

            interval = 1f / framesPerSecond;
        }

        //Check whether recording arguments are passed
        recordAgent |= ArgumentParser.Options.RecordAgent;
        recordChamber |= ArgumentParser.Options.RecordChamber;

        logDir = ArgumentParser.Options.LogDir;
        if (string.IsNullOrEmpty(logDir))
        {
            logDir = "./logfile/";
        }
        
        chamberDir = logDir + chamberRecorder.tag;

        if (!string.IsNullOrEmpty(chamberDir))
            Directory.CreateDirectory(chamberDir);
        
        agentDir = logDir + agentRecorder.tag;
        
        if (!string.IsNullOrEmpty(agentDir))
            Directory.CreateDirectory(agentDir);

    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= interval)
        {
            if (recordAgent)
            {
                Capture(agentRecorder, frame_counter, agentDir);
            }
            
            if (recordChamber)
            {
                Capture(chamberRecorder, frame_counter, chamberDir);
            }

            frame_counter++;
            timer = 0;

        }
        SwitchCameras();
    }

    void SwitchCameras()
    {
        int _previousCamera = _currentCamera;

        // Press arrow keys to switch between cameras.
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            _currentCamera = Mathf.Max(0, _currentCamera - 1);
        }
        else if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            _currentCamera = Mathf.Min(_cameras.Count - 1, _currentCamera + 1);
        }
        
        int currDisplay = _cameras[_currentCamera].targetDisplay;

        // Disable previous camera and enable current camera.
        if (_currentCamera != _previousCamera)
        {
            _cameras[_previousCamera].depth = 2;
            _cameras[_currentCamera].depth = 1;
            Debug.Log("switched cams");
        }
        
    }
    
    public void Capture(Camera camera, int frame_number, string dir)
    {
        //Temporary variable to store currently active texture
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = camera.targetTexture;

        camera.Render();

        Texture2D image = new Texture2D(camera.targetTexture.width, camera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, camera.targetTexture.width, camera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;

        byte[] bytes = image.EncodeToPNG();
        Destroy(image);

        File.WriteAllBytes($"{dir}/output_{frame_number}.png", bytes);
    }
}

