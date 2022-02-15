using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GrainSpawn : MonoBehaviour {

    public GameObject grainParticle;
    public int numberOfGrains;
    Renderer rend;
    Vector3 boundsMax;
    Vector3 boundsMin;
    public int grainSpawnSpeed = 10;

    private void Awake()
    {
        rend = GetComponent<Renderer>();
        boundsMax = new Vector3(rend.bounds.max.x, rend.bounds.max.y, rend.bounds.max.z);
        boundsMin = new Vector3(rend.bounds.min.x, rend.bounds.min.y, rend.bounds.min.z);
        StartCoroutine(SpawnFood(numberOfGrains, grainSpawnSpeed));
    }
    // Use this for initialization
    void Start () {

    }

    // Update is called once per frame
    void Update () {

    }

    IEnumerator SpawnFood(int particleNumber, int speed)
    {
        for (int i = 0; i < particleNumber; i = i + speed)
        {
            for (int j = 0; j < speed; j++)
            {
                Instantiate(grainParticle, new Vector3(Random.Range(boundsMin.x, boundsMax.x), boundsMax.y, Random.Range(boundsMin.z, boundsMax.z)), new Quaternion(0, 0, 0, 0));
            }
            yield return null;
        }
    }
}
