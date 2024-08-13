import torch
import numpy
import dstates.ai

CONNECTION_STRING="ofi+tcp://127.0.0.1:1234"

if __name__ == "__main__":
    # initialization
    torch.manual_seed(1729)
    backend = dstates.ai.evostore(CONNECTION_STRING.split('://')[0], [CONNECTION_STRING], 1 << 30)

    # save layers
    t1 = torch.rand(4, 5)
    t2 = torch.rand(2, 64)
    t3 = torch.rand(1, 20)
    lids = [0, 1, 2]
    edges = [0, 1, 1, 2]
    owners = [1, 1, 1]
    assert backend.save_layers([t1, t2, t3], 1, lids) == True
    assert backend.store_meta(1, edges, lids, owners, [80, 512, 80], 0.0) == True
    t4 = torch.rand(2, 64)
    lids = [0, 3, 2]
    edges = [0, 3, 3, 2]
    owners = [1, 2, 1]
    assert backend.save_layers([t4], 2, [3]) == True
    assert backend.store_meta(2, edges, lids, owners, [80, 512, 80], 0.0) == True

    # get longest prefix
    edges = [0, 3, 3, 1]
    (id, lids) = backend.get_prefix(edges)
    print("Model with longest prefix = %u, layer ids = %s" % (id, lids))
    assert len(lids) == 2

    # load layers
    t5 = torch.zeros(4, 5)
    t6 = torch.zeros(2, 64)
    t7 = torch.zeros(1, 20)
    comp = backend.get_composition(2)
    print("Composition of model_id 2 = %s" % comp)
    assert backend.load_layers([t5, t6, t7], 2, [0, 3, 2], [1, 2, 1]) == True

    # compare layers
    assert torch.equal(t1, t5) and torch.equal(t4, t6) and torch.equal(t3, t7)

    print("Success")
