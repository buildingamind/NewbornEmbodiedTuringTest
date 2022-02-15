using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Actuators;

public class ChickAgent : Agent
{
    [Tooltip("Distance the agent travels per unit time.")]
    public float MoveSpeed;

    [Tooltip("Angle of rotation per unit time.")]
    public float RotateSpeed;

    [Tooltip("Cost of moving forward.")]
    public float ForwardCost;

    [Tooltip("Cost of moving backward.")]
    public float BackwardCost;

    [Tooltip("Cost of rotation.")]
    public float RotateCost;

    [Tooltip("If true, spawn the agent in random position and orientation on reset.")]
    public bool TestMode;
    public float SpawnRange;
    public bool StickyWalls;

    Vector3 lastPosition;
    AgentLogger logger;
    FloatPropertiesChannel agentInfoChannel;


    public override void Initialize()
    {
        TestMode |= ArgumentParser.Options.TestMode;
        SetMaxStep();
        SetCameraResolution();
        SetLogger();

        // Register a side channel to communicate auxiliary agent information.
        agentInfoChannel = new FloatPropertiesChannel();
        if (ArgumentParser.Options.EnableAgentInfoChannel)
        {
            SideChannelManager.RegisterSideChannel(agentInfoChannel);
        }
    }

    private void SetMaxStep()
    {
        int maxStep = ArgumentParser.Options.EpisodeSteps;
        if (maxStep > 0)
        {
            this.MaxStep = maxStep;
        }
    }

    private void SetCameraResolution()
    {
        // Set input resolution before CameraSensors are initialized in Agent.OnEnable.
        int resolution = ArgumentParser.Options.InputResolution;
        var cameraSensorComponent = gameObject.GetComponent<CameraSensorComponent>();

        if (resolution > 0)
        {
            cameraSensorComponent.Height = resolution;
            cameraSensorComponent.Width = resolution;
        }
    }

    private void SetLogger()
    {
        var logDir = ArgumentParser.Options.LogDir;
        if (!System.String.IsNullOrEmpty(logDir))
        {
            logger = new AgentLogger(logDir, gameObject.name);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        lastPosition = transform.position;

        var moveDir = Vector3.zero;
        var rotateDir = Vector3.zero;
        var move = Mathf.FloorToInt(actions.DiscreteActions[0]);
        var rotate = Mathf.FloorToInt(actions.DiscreteActions[1]);

        switch (move)
        {
            case 0: // Do noting
                break;
            case 1: // W: Move forward
                moveDir = Vector3.forward * 1f;
                AddReward(ForwardCost);
                break;
            case 2: // S: Move backward
                moveDir = Vector3.forward * -1f;
                AddReward(BackwardCost);
                break;
        }

        switch(rotate)
        {
            case 0:
                break;
            case 1: // D: Turn right
                rotateDir = Vector3.up * 1f;
                AddReward(RotateCost);
                break;
            case 2: // A: Turn left
                rotateDir = Vector3.up * -1f;
                AddReward(RotateCost);
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * RotateSpeed);
        transform.Translate(moveDir * Time.deltaTime * MoveSpeed);

        // Send agent's position to python process.
        agentInfoChannel.Set("position_x", transform.position.x);
        agentInfoChannel.Set("position_y", transform.position.y);
        agentInfoChannel.Set("position_z", transform.position.z);

        // Log one timestep.
        if (logger != null)
        {
            logger.LogStep(
                CompletedEpisodes,
                StepCount,
                transform.position.x,
                transform.position.z,
                transform.rotation.y);
        }
    }

    public override void OnEpisodeBegin()
    {
        var academy = FindObjectOfType<ChickAcademy>();
        academy.ResetMonitors(CompletedEpisodes);

        if (TestMode)
        {
            transform.rotation = Quaternion.Euler(0, 0, 0);
            transform.position = new Vector3(0.0f, transform.position.y, 0.0f);
        }
        else
        {

            // Randomize position and rotation.
            transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
            transform.position = new Vector3(Random.Range(-SpawnRange, SpawnRange),
                                            transform.position.y,
                                            Random.Range(-SpawnRange, SpawnRange));
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        else
        {
            discreteActionsOut[0] = 0;
        }

        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 2;
        }
        else
        {
            discreteActionsOut[1] = 0;
        }
    }

    public void OnCollisionStay(Collision collision)
    {
        // Prevent sliding along the wall.
        if (collision.gameObject.tag == "Wall" && StickyWalls)
        {
            transform.position = lastPosition;
        }
    }
}
