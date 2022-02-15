using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class ChickAgentExternalReward : ChickAgent
{
    public override void OnActionReceived(ActionBuffers actions)
    {
        base.OnActionReceived(actions);
        AddReward(RayCastReward());
    }

    public float RayCastReward()
    {
        float reward = 0.0f;

        // Bit shift the index of the layer Target (9) to get a bit mask
        int layerMask = LayerMask.GetMask("ImprintTarget");

        RaycastHit hit;
        // Does the ray intersect any objects excluding the player layer
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out hit, 10, layerMask))
        {
            float dist_from_center = Mathf.Abs(hit.point.z - hit.transform.position.z);
            float dist_from_agent = hit.distance;
            reward = 0.2f * Mathf.Pow((1.0f / (1 + dist_from_agent)) * (1.0f / (1 + dist_from_center)), 2);
            //Debug.Log("Reward: " + reward.ToString());
        }
        //Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * 1000, Color.white);
        return reward;
    }
}
