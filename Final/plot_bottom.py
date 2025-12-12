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

    interpolated_point = ((inlet_points[0][1]-top_face[-1][1])/(inlet_points[0][0]-top_face[-1][0]))*(inlet_points[0][0]-diffuser_points[-1][0])+top_face[-1][1]

    if interpolated_point < diffuser_points[-1][1]:
        corner_point = (diffuser_points[-1][0], top_face[-1][1])
        length_of_front = np.sqrt((inlet_points[0][0] - corner_point[0])**2 + (inlet_points[0][1] - corner_point[1])**2)
        angle_of_front = np.rad2deg(np.arctan((inlet_points[0][1] - corner_point[1])/(inlet_points[0][0] - corner_point[0])))
        bottom_face.append(corner_point)

    bottom_face.append(top_face[-1])

    xs = [p[0] for p in top_face]
    ys = [p[1] for p in top_face]
    bx = [p[0] for p in bottom_face]
    by = [p[1] for p in bottom_face]

    plt.figure(figsize=(10,4))
    plt.plot(xs, ys)
    plt.plot(bx, by)
    plt.axis('equal')
    plt.show()

    return top_face, bottom_face, length_of_front, angle_of_front