import numpy as np
from matplotlib import pyplot as plt

def plot_bottom(inlet, diffuser_df, combustor_dict, x5s, h5s, x6s, h6s):
    # Top face
    top_face = []

    # Inlet
    inlet_points = [[x,y] for x,y in zip(inlet.xs, inlet.ys)]
    top_face.extend(inlet_points)
    
    prev_point = top_face[-1]
    prev_x = prev_point[0]
    prev_y = prev_point[1]

    # Diffuser
    diffuser_points = diffuser_df[['x', 'y']].values.tolist()
    for point in diffuser_points:
        point[0] += prev_x
        point[1] = -point[1] + (prev_y + diffuser_df['y'].iloc[0])
    top_face.extend(diffuser_points)

    print(f"x-diff: {top_face[-1][0]-prev_x}")

    prev_point = top_face[-1]
    prev_x = prev_point[0]
    prev_y = prev_point[1]

    print(f"prev_x: {prev_x}, prev_y: {prev_y}")
    print(f"combustor height: {combustor_dict['height_m']}")
    
    # Combustor
    combustor_point = (combustor_dict['length_m'] + prev_x, combustor_dict['height_m']+(prev_y-combustor_dict['height_m']))
    top_face.append(combustor_point)

    prev_point = top_face[-1]
    prev_x = prev_point[0]
    prev_y = prev_point[1]

    # Converging Section
    converging_section_points = [[x, y] for x, y in zip(x5s, -h5s)]
    for point in converging_section_points:
        point[0] += prev_x
        point[1] += (prev_y-(-h5s[0]))
    top_face.extend(converging_section_points)

    prev_point = top_face[-1]
    prev_x = prev_point[0]
    prev_y = prev_point[1]
    
    # Nozzle
    nozzle_points = [[x, y] for x, y in zip(x6s, -h6s)]
    for point in nozzle_points:
        point[0] += prev_x
        point[1] += (prev_y-(-h6s[0]))
    top_face.extend(nozzle_points)

    # Bottom face
    bottom_face = []
    bottom_face.append(inlet_points[0])
    bottom_face.append(top_face[-1])

    xs = [p[0] for p in top_face]
    ys = [p[1] for p in top_face]
    bx = [bottom_face[0][0], bottom_face[1][0]]
    by = [bottom_face[0][1], bottom_face[1][1]]

    plt.figure(figsize=(10,4))
    plt.plot(xs, ys)
    plt.plot(bx, by)
    plt.axis('equal')
    plt.show()

    return top_face, bottom_face