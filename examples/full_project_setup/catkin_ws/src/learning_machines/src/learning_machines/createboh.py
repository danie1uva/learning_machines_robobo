import pickle 
with open("examples/full_project_setup/catkin_ws/src/learning_machines/src/learning_machines/classes.txt", "r") as f:
    lines = f.read()
    print(lines)
    with open("examples/full_project_setup/catkin_ws/src/learning_machines/src/learning_machines/coco_classes.pickle", "wb") as f:
            classes = []
            for line in lines.split('\n'):
                    classes.append(line)
            print(classes)
            print('prima', len(classes))
            classes = classes[:len(classes)-1]
            print('dopo', len(classes))
            print(classes)
            pickle.dump(classes, f)