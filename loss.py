
def new_loss_function(p, y, b):
    eps = 0.000000000000000000000000001
    losses = 0
    k = len(y[0])
    # if torch.cuda.is_available():
    ones = torch.ones(1, 1).expand(b, k).cuda()
    # ones = torch.ones(1, 1).expand(b, k)
    loss1 = -((ones-y)*(((ones-p)+(eps)).log())).sum(dim=1)
    prod = (ones-y)*k - y*((p+ eps).log())
    loss2 = torch.min(prod, dim=1)[0]
    losses = (loss1 + loss2).sum()
    return losses / b