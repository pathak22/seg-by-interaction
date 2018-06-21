## Learning Instance Segmentation by Interaction ##
#### [[Project Website]](https://pathak22.github.io/seg-by-interaction/) [[Videos]](http://pathak22.github.io/seg-by-interaction/index.html#demoVideos)

Deepak Pathak\*, Yide Shentu\*, Dian Chen\*, Pulkit Agrawal\*, Trevor Darrell, Sergey Levine, Jitendra Malik<br/>
University of California, Berkeley<br/>

<img src="https://pathak22.github.io/seg-by-interaction/resources/teaser.jpg" width="300">

This is the implementation for the paper on [Learning Instance Segmentation by Interaction](https://pathak22.github.io/seg-by-interaction). We present an approach for building an active agent that learns to segment its visual observations into individual objects by interacting with its environment in a completely self-supervised manner. The agent uses its current segmentation model to infer pixels that constitute objects and refines the segmentation model by interacting with these pixels.

    @inproceedings{pathakArxiv18segByInt,
        Author = {Pathak, Deepak and
        Shentu, Yide and Chen, Dian and
        Agrawal, Pulkit and Darrell, Trevor and
        Levine, Sergey and Malik, Jitendra},
        Title = {Learning Instance Segmentation by Interaction},
        Booktitle = {arXiv},
        Year = {2018}
    }

### 1) Robot Interaction Dataset

To be released soon. Email me in case you need early access.

### 2) Robust Set Loss

We propose a technique, "robust set loss", to handle noisy segmentation training signal, with the general idea being that the segmenter is not required to predict exactly the pixels in the candidate object mask, rather that the predicted pixels as a set have a good Jaccard index overlap with the candidate mask. We show that robust set loss significantly improves segmentation performance and also reduces the variance in results.

Implementation in file: `seg-by-interaction/robust_set_loss.py`
