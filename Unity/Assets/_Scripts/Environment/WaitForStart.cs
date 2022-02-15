using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaitForStart : MonoBehaviour {
    Camera cam;
    int numberOfGrains = 2000;
    int oldMask;

    // Use this for initialization
    IEnumerator Start () {
        cam = GetComponent<Camera>();
        oldMask = cam.cullingMask;
        cam.cullingMask = (1 << LayerMask.NameToLayer("Black"));
        StartCoroutine(CountFood());
        yield return new WaitForSeconds(2f);
        Debug.Log("Done");
        cam.cullingMask = oldMask;
    }

    // Update is called once per frame
    void Update () {

    }

    IEnumerator CountFood()
    {
        if (GameObject.FindGameObjectsWithTag("food").Length < numberOfGrains)
        {
            yield return null;
        }
    }
}
