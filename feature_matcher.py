import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class FeatureMatcher(ABC):
    def __init__(self, **params):
        self.params = params
        self._init_matcher()
    
    def __str__(self):
        return self.__class__.__name__.replace('FeatureMatcher', '')

    @abstractmethod
    def _init_matcher(self):
        """Инициализация детектора и дескриптора особенностей"""
        pass
        
    def estimate_transform(self, kp1: np.ndarray, kp2: np.ndarray, matches: Any) -> np.ndarray:
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
        return M
    
    def process_images(self, img_ideal: np.ndarray, img_input: np.ndarray) -> Dict[str, Any]:
        kp1, desc1 = self.detect_and_compute(img_ideal)
        kp2, desc2 = self.detect_and_compute(img_input)
        
        matches = self.match_features(desc1, desc2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        M = self.estimate_transform(kp1, kp2, matches)
        
        return {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': matches,
            'matrix_alg': M
        }
    
    def detect_and_compute(self, image):
        return self.detector.detectAndCompute(image, None)
    
    def match_features(self, desc1, desc2):
        return self.matcher.match(desc1, desc2)

class SIFTFeatureMatcher(FeatureMatcher):
    def _init_matcher(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
class FASTBRIEFFeatureMatcher(FeatureMatcher):
    def __str__(self):
        return 'FAST+BRIEF'
          
    def _init_matcher(self):
        self.detector = cv2.FastFeatureDetector_create()
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image):
        kp = self.detector.detect(image, None)
        kp, desc = self.descriptor.compute(image, kp)
        return kp, desc
    
    def match_features(self, desc1, desc2):
        return self.matcher.match(desc1, desc2)

class ORBFeatureMatcher(FeatureMatcher):
    def _init_matcher(self):
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
