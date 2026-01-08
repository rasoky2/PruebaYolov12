
# Colores Shadcn Light (Zinc based)
STYLES = """
QMainWindow, QWidget {
    background-color: #FFFFFF;
    color: #09090b;
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* Tipografía */
QLabel {
    color: #09090b;
    font-size: 13px;
}
QLabel#Title {
    font-size: 16px;
    font-weight: bold;
    color: #18181b;
}
QLabel#Subtitle {
    color: #71717a;
    font-size: 12px;
}

/* Botones (Primary) */
QPushButton {
    background-color: #18181b;
    color: #fafafa;
    border: 1px solid #18181b;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #27272a; /* Zinc 800 */
}
QPushButton:pressed {
    background-color: #3f3f46;
}
QPushButton:disabled {
    background-color: #e4e4e7;
    color: #a1a1aa;
    border: 1px solid #e4e4e7;
}

/* Botones Secundarios (Ghost/Outline style for others) */
QPushButton#Secondary {
    background-color: #FFFFFF;
    color: #18181b;
    border: 1px solid #e4e4e7;
}
QPushButton#Secondary:hover {
    background-color: #f4f4f5;
}

/* Botones Destructivos */
QPushButton#Destructive {
    background-color: #ef4444;
    border: 1px solid #ef4444;
}
QPushButton#Destructive:hover {
    background-color: #dc2626;
}

/* Inputs y Combos */
QComboBox, QSpinBox, QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    padding: 6px;
    color: #09090b;
    selection-background-color: #f4f4f5;
    selection-color: #18181b;
}
QComboBox:hover, QSpinBox:hover {
    border: 1px solid #a1a1aa;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}

/* GroupBox y Frames */
QGroupBox {
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
    color: #18181b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #FFFFFF;
}

QFrame#Card {
    background-color: #FFFFFF;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
}

/* Logs */
QTextEdit {
    background-color: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    color: #18181b;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 12px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #f4f4f5;
    width: 8px;
    margin: 0px 0px 0px 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #d4d4d8;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QSplitter::handle {
    background-color: #e4e4e7;
}
"""
