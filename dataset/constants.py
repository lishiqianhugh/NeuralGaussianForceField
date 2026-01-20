import torch

OBJ_INDEX = {'ball': 0, 'bear': 1, 'bowl': 2, 'can': 3, 'cloth': 4, 'cloth2': 5, 'cloth3': 6, 'duck': 7, 'duck2': 8, 'miku': 9, 'panda': 10, 'phone': 11, 'pillow': 12, 'pillow2': 13, 'rope': 14, 'rope2': 15, 'soccer': 16, 'toy': 17}
INDEX_OBJ = {0: 'ball', 1: 'bear', 2: 'bowl', 3: 'can', 4: 'cloth', 5: 'cloth2', 6: 'cloth3', 7: 'duck', 8: 'duck2', 9: 'miku', 10: 'panda', 11: 'phone', 12: 'pillow', 13: 'pillow2', 14: 'rope', 15: 'rope2', 16: 'soccer', 17: 'toy'}
OBJPART = {'ball': 106115, 'bear': 139018, 'bowl': 232764, 'can': 70569, 'cloth': 116188, 'cloth2': 37695, 'cloth3': 253341, 'duck': 80544, 'duck2': 8326, 'miku': 48316, 'panda': 56104, 'phone': 26571, 'pillow': 64229, 'pillow2': 61882, 'rope': 8051, 'rope2': 104410, 'soccer': 54547, 'toy': 60283}
PRIOR_SCALE = {0:0.4, 1:0.65, 2:0.5, 3:1.0, 4:1.0, 5: 1.0, 6: 0.6, 7:0.3, 8:0.65, 9: 0.6, 10: 0.6, 11:0.4, 12:0.7, 13: 1.2, 14:1.0, 15:0.65, 16:1.0, 17:1.0}

TRAIN_IDS = [0, 2, 3, 4, 7, 9, 10, 11, 12, 14]
TEST_IDS = list(INDEX_OBJ.keys() - set(TRAIN_IDS))

INDEX_OBJ = {i: INDEX_OBJ[i] for i in TRAIN_IDS}
OBJ_INDEX = {v: k for k, v in INDEX_OBJ.items()}
OBJPART   = {name: OBJPART[name] for name in INDEX_OBJ.values()}
PRIOR_SCALE = {i: PRIOR_SCALE[i] for i in TRAIN_IDS}

INDEX_OBJ_TEXT = {
        0:  "A pokemon ball.",
        1:  "A teddy bear toy.",
        2:  "A blue bowl with cloud.",
        3:  "A cola.",
        4:  "A yellow t-shirt with a owl on it.",
        5:  "A checkered cloth.",
        6:  "A cyan mouse pad.",
        7:  "A rubber duck.",
        8:  "A yellow duck toy.",
        9:  "A hatsune miku.",
        10: "A cartoon panda.",
        11: "A mobile phone.",
        12: "A blue and white stripped pillow.",
        13: "A black and white stripped pillow.",
        14: "A thick rope.",
        15: "A rope.",
        16: "A soccer ball.",
        17: "A Roujiamo toy."
    }

SCENES = {
    "table": { # black and white
        "rotation": [-113.2965, 3.5640, 63.4301],
        "translation": [-0.5443, -0.10, 0.5878],
        "scale": 1.878
    },
    "table0": { # cucumber 151
        "rotation": [-115, 2.0, 63.4301],
        "translation": [0.1, -0.2, 0.95],
        "scale": 3.3
    },
    "table1": { # green and blue table like sea
        "rotation": [-115, 2.0, 63.4301],
        "translation": [-0.5, 0.0, 0.65],
        "scale": 3.0
    },
    "table2": { # green and blue table like sea
        "rotation": [-115, 2.0, 63.4301],
        "translation": [0.0, 0.6, 0.45],
        "scale": 2.2
    },
    "table3": { # purple with words
        "rotation": [-116.0, 3.0, 60.0],
        "translation": [-0.3, 0.8, 0.45],
        "scale": 2.2
    },
    "table4": { # cucumber 151
        "rotation": [-115, 2.0, 63.4301],
        "translation": [0.1, -0.2, 1.05],
        "scale": 3.5
    },
    "table5": { # table with no cloth
        "rotation": [-113.2965, 3.5640, 63.4301],
        "translation": [0.0, 0.0, 0.7],
        "scale": 2.2
    },
    "table6": { # white neuron-texture black table
        "rotation": [-113, 4.0, 63.4301],
        "translation": [-0.5, 0.0, 0.75],
        "scale": 3.0
    },
    "table7": { # green and blue table like sea
        "rotation": [-115, 2.0, 63.4301],
        "translation": [-0.5, 0.6, 0.62],
        "scale": 2.5
    }
}

def generate_ground_plane(x_range, y_range, num_points_x, num_points_y, z_value, frame_num,
                                square_size=0.2, light_opacity=0.0, dark_opacity=0.4,
                                device='cuda'):
    x = torch.linspace(x_range[0], x_range[1], num_points_x, device=device)
    y = torch.linspace(y_range[0], y_range[1], num_points_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z_value)

    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    num_points = points.shape[0]

    positions = points.unsqueeze(0).repeat(frame_num, 1, 1)  # [frames, N, 3]


    ix = ((xx - x_range[0]) / square_size).floor().long()
    iy = ((yy - y_range[0]) / square_size).floor().long()
    checker_mask = (ix + iy) % 2

    checker_mask = checker_mask.reshape(-1)

    covariances = torch.tensor([3e-4, 0.0, 0.0,
                                3e-4, 0.0,
                                1e-6], device=device).view(1, 1, 6).repeat(frame_num, num_points, 1)

    rotations = torch.eye(3, device=device).unsqueeze(0).repeat(num_points, 1, 1)

    opacities = torch.where(checker_mask.unsqueeze(-1) == 0,
                            torch.full((num_points, 1), light_opacity, device=device),
                            torch.full((num_points, 1), dark_opacity, device=device))

    shs = torch.zeros((num_points, 9, 3), device=device)

    return {
        'positions': positions,
        'covariances': covariances,
        'rotations': rotations,
        'opacities': opacities,
        'shs': shs
    }
