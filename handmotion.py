import tensorflow as tf
import numpy as np

class HandMotion():
    def __init__(self, class_names, modelfile):
        self.handmotion = np.zeros((50, 42))
        self.interprefer = tf.lite.Interpreter(model_path=modelfile)
        self.interprefer.allocate_tensors()
        self.input_details = self.interprefer.get_input_details()
        self.output_details = self.interprefer.get_output_details()
        # self.class_names = ['BYE', 'COMEON', 'PUSH', 'STONEPAPER', 'SWIPE']
        self.class_names = class_names

    def input(self, hands):
        for hand in hands:
            self.handmotion = np.delete(self.handmotion, 0, 0)
            self.handmotion = np.concatenate([self.handmotion, np.concatenate([np.array(hand.landmarks_resized)[:,0].reshape(21), np.array(hand.landmarks_resized)[:,1].reshape(21)], axis=0).reshape(1, 42)], axis=0)

    def predict(self):
        self.interprefer.set_tensor(self.input_details[0]['index'], self.handmotion.reshape(1, 50, 42).astype(np.float32))
        self.interprefer.invoke()
        output_data = self.interprefer.get_tensor(self.output_details[0]['index'])
        return self.class_names[np.argmax(output_data)]

    def run(self, hands):
        self.input(hands)
        return self.predict()

