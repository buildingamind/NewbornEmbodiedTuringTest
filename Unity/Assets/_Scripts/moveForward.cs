using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class moveForward : MonoBehaviour
{
    public float rotationSpeed = 1;
    public NavMeshAgent agent;
    public GameObject[] toSee;
    private int seeIndex = 0;
    public int stareTime = 100;
    [Range(0.0f, 1.0f)]
    public float fractionStare = 0.8f;
    private int counter = 0;
    private Vector3 lastPosition;
    private Vector3 randLoc;
    private bool looking = true;

    // Start is called before the first frame update
    void Start()
    {
	    PickDestination(toSee);
	    lastPosition = this.transform.position;
        randLoc = Random.insideUnitSphere * 5 + Vector3.up;
    }

    // Update is called once per frame
    void Update()
    {
        GameObject[] targets = toSee;
        
        if (counter == 0)
        {
            randLoc = Random.insideUnitSphere * 5 + Vector3.up;
        }

        if (Random.Range(0.0f, 1.0f) > fractionStare && !looking)
        {
            RandomNav(randLoc);
            return;
        }

        LookAt(toSee);
    }

    private void PickDestination(GameObject[] targets){
        agent.destination = targets[seeIndex].transform.position;
    }

    private void LookAt(GameObject[] targets)
    {
        looking = true;
	    Vector3 relativePos = targets[seeIndex].transform.position - transform.position;
        relativePos[1] = 0;
        Quaternion toRotation = Quaternion.LookRotation(relativePos);
        transform.rotation = Quaternion.Lerp( transform.rotation, toRotation, 1 * Time.deltaTime );
        if (Vector3.Distance(lastPosition, this.transform.position) <= 0.01){
            counter++;
        }
        if (counter >= stareTime){
            counter = 0;
            seeIndex++;
            seeIndex %= targets.Length;
            PickDestination(targets);
            looking = false;
        }
        lastPosition = this.transform.position;
    }
    
    private void RandomNav(Vector3 target)
    {
	    Vector3 relativePos = target - transform.position;
        relativePos[1] = 0;
        Quaternion toRotation = Quaternion.LookRotation(relativePos);
        transform.rotation = Quaternion.Lerp(transform.rotation, toRotation, 1 * Time.deltaTime );
        agent.destination = target;
        lastPosition = this.transform.position;
        counter++;
        
        if (counter >= stareTime){
            counter = 0;
        }
    }
}
