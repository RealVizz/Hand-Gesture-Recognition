import cv2
import numpy as np
import pickle

def predict_image(data, clf):
    n_samples = len(data)
    clean_data = data.flatten().reshape(1, -1)
    clean_data = clean_data / 255
    pred = clf.predict(clean_data)
    print(pred)
    return pred

def load_model(name):
    with open(name, mode="rb") as f:
        model = pickle.load(f)
        return model

def run():
    subtractor = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture(0)
    clf = load_model("handmodel.model")

    try:
        while True:
            ret, frame = cap.read()

            fgmask = subtractor.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)

            target = cv2.resize(fgmask[0 : 300, 0 : 300], (150, 150))
            prediction = predict_image(target, clf)

            # Code taken from https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
            cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 3)
            if prediction is not None:
                cv2.putText(frame, str(prediction[0]), (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Input for Model", target)
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask', fgmask)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(e)
        cap.release()
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()

