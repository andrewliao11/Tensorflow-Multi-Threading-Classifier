# Multi-Threading-mnist-classifier
The project aims at implementing a simple mnist classifer with **multi-thread FIFOQueue**

## dependencies
- python2
- tensorflow (>0.12)
- cuda (>8.0)
- other requirements
```
pip install --user -r requirements.txt
```

## usage
Available options include:
```
--lr            (default 3e-4, initial learning rate)
-- batch_size   (default 128, batch_size)
```
To run the model:
```
python main.py [args]
```   
**Note: the result show on the tensorboard/terminal is training loss/accuracy**

## how this works?

## reference
- [TensorFlow Data Input (Part 1): Placeholders, Protobufs & Queues](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)

