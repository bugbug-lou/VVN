import tensorflow.compat.v1 as tf

def collect_and_flatten(inputs, outputs, targets, **kwargs):
    print("targets in collect and flatten", targets)
    target_outputs = {}
    for target in targets:
        if target in outputs.keys():
            if isinstance(outputs[target], dict):
                for t in outputs[target]:
                    target_outputs[t] = outputs[target][t]
            else:
                target_outputs[target] = outputs[target]
    return target_outputs

def total_loss(logits, labels, **kwargs):

    loss = tf.constant(0.0, tf.float32)
    assert all((isinstance(loss, tf.Tensor) for loss in logits.values()))
    for loss_name, loss_val in logits.items():
        print("Using loss %s" % loss_name)
        loss += loss_val

    return loss
