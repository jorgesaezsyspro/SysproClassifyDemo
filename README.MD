Demo code to classify a image taken from a camera using a Deep Learning model.
The output will be send to a PLC to take some actions.

Train images should be organized by class in a folder named accordantly.
Test images doesn´t need to be organized.

Test dataset:
    https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

E.g.:

    ./imgs
        |- train
            |- threads
                |- th1.png
                |- th2.png
                |. ...
            |- nuts
                |- nu1.png
                |- nu2.png
                |. ...
            |- screw
            |- bolt
        |- test
            |- test1.png
            |- test2.png
