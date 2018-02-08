# This fork dedicates on training the models for TFO
## The following are the training steps and the corresponding files for a training example.

<b>1.Prepare the Data:</b> [prepare_data_example.ipynb](https://github.com/GBJim/tfo_training/blob/master/research/object_detection/prepare_data_example.ipynb) 

<b>2.Prepare the Pipeline Congifuration:</b> [example_fastrrcnn_inception.config](https://github.com/GBJim/tfo_training/blob/master/research/object_detection/config/example_fastrrcnn_inception.config)

<b>3. Start the Training Process:</b> [training_example.py](https://github.com/GBJim/tfo_training/blob/master/research/object_detection/training_example.py)
```Shell
#Make sure you are at models/research/
python object_detection/training_example.py
```

4. <b>Export the Trained Model for Inference:</b>[export_inference_graph.py](https://github.com/GBJim/tfo_training/blob/master/research/object_detection/export_inference_graph.py), please refer to [this document](https://github.com/GBJim/tfo_training/blob/master/research/object_detection/g3doc/exporting_models.md) for details.
