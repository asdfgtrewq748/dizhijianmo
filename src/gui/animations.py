from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QAbstractAnimation, QParallelAnimationGroup, QPoint, QRect
from PyQt6.QtWidgets import QWidget, QGraphicsOpacityEffect

class AnimationUtils:
    @staticmethod
    def fade_in(widget: QWidget, duration: int = 500):
        """Fade in a widget."""
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        
        # Keep reference to avoid garbage collection
        widget._fade_animation = animation 

    @staticmethod
    def slide_in(widget: QWidget, start_pos: QPoint, end_pos: QPoint, duration: int = 500):
        """Slide a widget from start_pos to end_pos."""
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        animation.setStartValue(start_pos)
        animation.setEndValue(end_pos)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        
        widget._slide_animation = animation

    @staticmethod
    def shake(widget: QWidget):
        """Shake animation for error indication."""
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(500)
        animation.setLoopCount(1)
        
        start_pos = widget.pos()
        
        # Create keyframes
        key_frames = []
        for i in range(0, 10):
            offset = 5 if i % 2 == 0 else -5
            key_frames.append((i / 10.0, QPoint(start_pos.x() + offset, start_pos.y())))
        key_frames.append((1.0, start_pos))
        
        for step, value in key_frames:
            animation.setKeyValueAt(step, value)
            
        animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        widget._shake_animation = animation
