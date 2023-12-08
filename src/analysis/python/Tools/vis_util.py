from mlagents.tf_utils import tf
import numpy as np

from lucid.modelzoo.vision_models import Model

# Adapted from https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
def load_frozen_graph(path, verbose=False):
    graph = tf.Graph()
    with tf.gfile.GFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    graph.finalize()

    return graph


def save_as_lucid_model(graph, save_path=None):
    if not save_path:
        save_path = 'lucid_model.pb'

    # Infer save args from graph
    input_name = 'visual_observation_0'
    output_names = [op.name for op in graph.get_operations()] # Save all layers.
    image_shape = graph.get_operation_by_name(input_name).outputs[0].shape[1:].as_list()

    with tf.Session(graph=graph) as sess:
        Model.save(
            save_path,
            input_name=input_name,
            image_shape=image_shape,
            output_names=output_names,
            image_value_range=[0,1],
        )


def load_model(graph_path):
    """ Load frozen graph as Lucid Model. """
    graph = load_frozen_graph(graph_path)
    save_as_lucid_model(graph, 'lucid_model.pb')
    model = Model.load('lucid_model.pb')
    output_names = [op.name for op in graph.get_operations()]
    return model, output_names

