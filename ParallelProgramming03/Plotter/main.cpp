#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    // Set environment variables to avoid warnings
    qputenv("QT_DEVICE_PIXEL_RATIO", "0");
    qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "1");
    qputenv("QT_SCREEN_SCALE_FACTORS", "1");
    qputenv("QT_SCALE_FACTOR", "1");

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
