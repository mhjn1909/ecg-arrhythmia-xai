import torch


def compute_integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """
    Integrated Gradients implementation
    """

    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(input_tensor.device)

    # Generate scaled inputs
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    grads = []

    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_()

        output = model(scaled_input)
        score = output[:, target_class]

        model.zero_grad()
        score.backward()

        grads.append(scaled_input.grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    integrated_gradients = (input_tensor - baseline) * avg_grads

    return integrated_gradients.abs().squeeze().cpu().numpy()

