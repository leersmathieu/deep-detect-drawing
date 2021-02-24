
# Deep Detect Drawing

Deep detect Drawing is a app based on a CNN model, to recognize a few draw.
You can check the model construction [here](https://github.com/leersmathieu/deep-detect-drawing/blob/master/modeling%20notebook/object_recognition_model.ipynb)

# Training & accuracy

![train](assets/train.png)
# Try it yourself !

On my serveur http://deepdrawing.tamikara.xyz/

OR

With *docker*


For try this app on your computer with docker, just enter this command in your terminal.   
It's magic.

```docker
docker run -p 8501:8501 leersma/deep-detect-drawing:latest
```
just pay attention to the port used
If you run it on port 8501 as in the example, you can easily access it like this: http://localhost:8501/.
