from threading import Lock, Thread
import cv2


class CamReader:
    def __init__(
        self, url, start_frame=0, color_order="BGR", buffer_size=2, loop=False
    ):
        self._is_rtsp = False
        self.loop = loop
        self.offset = 0
        assert color_order in {
            "BGR",
            "RGB",
        }, "color_order must be 'BGR' (OpenCV default) or 'RGB'"
        self.color_order = color_order

        if url.startswith("rtsp://") or url.startswith("http://"):
            self._is_rtsp = True
            self.reader = ThreadedCamReader(url, buffer_size)
            self.reader.start()
            self.frame_count = -1
            self.fps = self.reader.cap.get(cv2.CAP_PROP_FPS)
            self.width = self.reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            self.reader = cv2.VideoCapture(url)
            self.frame_count = self.reader.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = self.reader.get(cv2.CAP_PROP_FPS)
            self.width = self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if start_frame is not None and start_frame != 0:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.next_frame_index = 0 - self.offset

    def stop(self):
        if self._is_rtsp:
            self.reader.stop()
            self.reader.join()
        else:
            self.reader.release()

    def goto_frame(self, idx):
        if self._is_rtsp:
            print("Invalid action on RTSP stream")
            return

        new_idx = idx - self.offset
        if new_idx == self.next_frame_index:
            return

        if idx == -1:
            new_idx = self.frame_count - 1

        if new_idx > self.frame_count - 1:
            new_idx = self.frame_count - 1

        self.next_frame_index = new_idx
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, max(0, new_idx))

    def get_next_frame_idx(self):
        return self.next_frame_index + self.offset

    def get_image(self):
        if self._is_rtsp:
            ret, image, frame_idx = self.reader.get_image()
            self.next_frame_index = frame_idx + 1
        else:
            if self.next_frame_index == self.frame_count:
                if not self.loop:
                    return False, None, None
                else:
                    self.goto_frame(self, 0)
                    return self.get_image()

            frame_idx = self.get_next_frame_idx()
            if self.next_frame_index <= 0:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, image = self.reader.read()
            self.next_frame_index += 1

        if ret and image is not None and self.color_order == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return ret, image, frame_idx


class ThreadedCamReader(Thread):
    def __init__(self, url, buffer_size=2):
        Thread.__init__(self)
        self.is_running = True
        self.frame_idx = -1
        self.cap = cv2.VideoCapture(url)#, cv2.CAP_FFMPEG)
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx_to_write = 0
        self.idx_to_read = 1
        self.lock = Lock()

    def stop(self):
        self.is_running = False

    def get_image(self):
        with self.lock:
            frame = self.buffer[self.idx_to_read]
            frame_idx = self.frame_idx
            if frame is not None:
                frame = frame.copy()
        return frame is not None, frame, frame_idx

    def run(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            self.frame_idx += 1
            
            if ret:
                with self.lock:
                    self.buffer[self.idx_to_write] = frame
                    self.idx_to_write = (self.idx_to_write + 1) % self.buffer_size
                    self.idx_to_read = (self.idx_to_read + 1) % self.buffer_size
            
        self.cap.release()
