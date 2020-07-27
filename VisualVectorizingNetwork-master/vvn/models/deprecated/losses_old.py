def end_to_end_loss(
        logits,
        labels,
        encoder_loss_scale = 1.0,
        selfsup_loss_scale = 1.0,
        render_loss_scale = 1.0,
        spatial_loss_scale = 1.0,
        future_loss_scale = 1.0,
        shapes_loss_scale = 1.0,
        distribution_loss_scale = 1.0,
        relative_attr_loss_scale = 1.0,
        vae_loss_scale= 1.0,
        feature_loss_scale = 1.0,
        agent_loss_scale = 1.0,
        dynamics_loss_scale = 1.0,
        hrn_loss_scale = 1.0,
        imagenet_loss_scale = 0
        ):
    """
    Wrapper to combine particle encoder loss with hrn loss.
    """
    loss = 0.0

    # Encoder loss
    if 'encoder_loss' in logits and encoder_loss_scale > 0.0:
        print("Using Encoder loss scaled by %f" % encoder_loss_scale)
        loss += logits['encoder_loss']['encoder_loss'] * encoder_loss_scale

    # Feature loss
    if 'feature_loss' in logits and feature_loss_scale > 0.0:
        print("Using Feature loss scaled by %f" % feature_loss_scale)
        loss += logits['feature_loss']['feature_loss'] * feature_loss_scale

    # Selfsup loss
    if 'selfsup_loss' in logits and selfsup_loss_scale > 0.0:
        print("Using Selfsupervised loss scaled by %f" % selfsup_loss_scale)
        loss += logits['selfsup_loss']['selfsup_loss'] * selfsup_loss_scale

    # rendering losst
    if 'render_loss' in logits and render_loss_scale > 0.0:
        print("Using rRentder loss scaled by %f" % render_loss_scale)
        loss += logits['render_loss']['render_loss'] * render_loss_scale

    # spatial attr lossa
    if 'spatial_loss' in logits and spatial_loss_scale > 0.0:
        print("Using Spatial Decoding loss scaled by %f" % spatial_loss_scale)
        loss += logits['spatial_loss']['spatial_loss'] * spatial_loss_scale

    if 'future_loss' in logits and future_loss_scale > 0.0:
        print("Using Future Decoding Loss scaled by %f" % future_loss_scale)
        loss += logits['future_loss']['future_loss'] * future_loss_scale

    if 'shapes_loss' in logits and shapes_loss_scale > 0.0:
        print("Using Shapes Decoding Loss scaled by %f" % shapes_loss_scale)
        loss += logits['shapes_loss']['shapes_loss'] * shapes_loss_scale

    # spatial attr lossa
    if 'distribution_loss' in logits and distribution_loss_scale > 0.0:
        print("Using Distribution loss scaled by %f" % distribution_loss_scale)
        loss += logits['distribution_loss']['distribution_loss'] * distribution_loss_scale

    # relative attr loss
    if 'relative_attr_loss' in logits and relative_attr_loss_scale > 0.0:
        print("Using Relative Attr loss scaled by %f" % relative_attr_loss_scale)
        loss += logits['relative_attr_loss']['relative_attr_loss'] * relative_attr_loss_scale

    # edge vae loss
    if (logits.get('vae_loss', None) is not None) and vae_loss_scale > 0.0:
        print("Using VAE loss scaled by %f" % vae_loss_scale)
        loss += logits['vae_loss']['vae_loss'] * vae_loss_scale

    # agent loss
    if 'agent_loss' in logits and agent_loss_scale > 0.0:
        print("Using Agent loss scaled by %f" % agent_loss_scale)
        loss += logits['agent_loss']['agent_loss'] * agent_loss_scale

    # dynamics loss
    if 'dynamics_loss' in logits and dynamics_loss_scale > 0.0:
        print("Using Dynamics loss scaled by %f" % dynamics_loss_scale)
        loss += logits['dynamics_loss']['dynamics_loss'] * dynamics_loss_scale

    # HRN loss
    if 'hrn_loss' in logits and hrn_loss_scale > 0.0:
        print("Using HRN loss scaled by %f" % hrn_loss_scale)
        loss += logits['hrn_loss']['hrn_loss'] * hrn_loss_scale

    # imagenet loss
    if 'imagenet_loss' in logits and imagenet_loss_scale > 0.0:
        print("Using Imagenet loss scaled by %f" % imagenet_loss_scale)
        loss += logits['imagenet_loss']['imagenet_loss'] * imagenet_loss_scale

    return loss
