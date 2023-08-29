import torch.nn.functional as F

def test(model, data, writer=None):
    model.eval()
    print("Final test of the model")
    output = model(data.x, data.edge_index)
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    if writer is not None:
        global_step = len(data) + 1
        writer.add_scalar('Loss/test', loss_test.item(), global_step)
