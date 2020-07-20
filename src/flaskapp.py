
from flask import Flask, request, render_template
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

app = Flask(__name__)

#To launch home page
@app.route('/')
def home():
    return "hello"



@app.route('/generate_unconditional',methods=['POST'])
def generate_unconditional(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        print("output",output)
        generated = 0
        raw_text=request.form.get('text')
        try:
            nsamples=int(request.form.get("count"))
        except:
            print("Exception")
        context_tokens = enc.encode(raw_text)
        generated = 0
        li=[]
        for _ in range(nsamples):
            out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]})[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                li.append(text)
                print(text)
                print("=" * 80)
        res={"data":li,"message":"Request Successful","statusCode":200}
    return res
    
if __name__ == "__main__":
    app.run()
