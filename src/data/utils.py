import os

def get_nodes_edges(datadir):
    """
    Return filenames with nodes and edges in a directory.
    :param datadir:
    :return:
    """

    filenames = os.listdir(datadir)


    filenames_coords = set()
    filenames_edges = set()
    for fn in filenames:
        if 'boundary' in fn or 'HI.txt' in fn:
            pass
        else:
            if 'COORDS' in fn:
                filenames_coords.add(fn)
            elif 'NCOL' in fn:
                filenames_edges.add(fn)
            else:
                print(fn)

    return filenames_coords, filenames_edges



