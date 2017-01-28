# Multi-Threading-mnist-classifier
The project aims at implementing a simple mnist classifer with **multi-thread FIFOQueue**

## what's the advantage of using FIFOQueue?
For normal supervised learning coding style, the mainly pipeline may looks like:   
```python
for iter in range(max_iters):
  inputs, labels = data_loader.load_next_batch()
  feed_dict = {inputs_tf:inputs, labels_tf:labels}
  _, summary_str = sess.run([train_op, summary_op], feed_dict)
  writer.add_summary(summary_str, iter)
  if iter%eval_per_iters == 0:
    eval(......)
```

This is fairly straightforward and easy to implement. However, its main drawback is: when the model is loading data to memory, the gpu is hanging there, which slower the training process :sweat:   
tf.FIFOQueue provides us another way to fully utilize the computation resources. In the previous pipeline, the model will **sequentially** load(CPU) the data to memory and then do the update(GPU). What if we can do the update and simultaneously prepare the next batch?

## how this works?
1. construct a binary file to load your save the data into the tensorflow format
  - use ```tf.python_io.TFRecordWriter```(recommended), which containing tf.train.Example protocol buffers (which contain Features as a field).
  - this step construct a **binary file** in the specified path
  - example code in ```construct_binary.py```
2. read binary file as tensor
  - use ```tf.TFRecordReader```(recommended) and decode the binary file as the way you encode the binary file(step 1)
  - this step will return your encode data(eg. inputs, labels). note that the return tensor represent a single tensor
  - if you call ```sess.run([inputs, labels])```, the command will always return the next pair.
  - example code in ```reader.py```
3. create batch
  - use ```tf.train.shuffle_batch``` to create batch from the return single paired data
4. initialize the dataflow graph
  - with command ```tf.train.start_queue_runners(sess=sess)```

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

## reference
- [TensorFlow Data Input (Part 1): Placeholders, Protobufs & Queues](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)



