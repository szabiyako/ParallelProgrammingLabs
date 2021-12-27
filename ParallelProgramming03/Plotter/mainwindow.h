#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QInputDialog>
#include "qcustomplot.h"
#include <QPointF>

#include <QProcess>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    virtual void closeEvent(QCloseEvent *event) override;

private slots:
    void titleDoubleClick(QMouseEvent *event);
    void axisLabelDoubleClick(QCPAxis* axis, QCPAxis::SelectablePart part);
    void legendDoubleClick(QCPLegend* legend, QCPAbstractLegendItem* item);
    void selectionChanged();
    void mousePress();
    void mouseWheel();
    void addRandomGraph();
    void removeSelectedGraph();
    void removeAllGraphs();
    void contextMenuRequest(QPoint pos);
    void moveLegend();
    void graphClicked(QCPAbstractPlottable *plottable, int dataIndex);

    void loadGraphs(
            const size_t minSide,
            const size_t maxSide,
            const int iters,
            const int step,
            const bool loadCpu,
            const bool loadStatic,
            const bool loadDynamic,
            const bool loadGuided);
    void updateGraphs(
            const QVector<double> &xCpu,
            const QVector<double> &yCpu,
            const QVector<double> &xStatic,
            const QVector<double> &yStatic,
            const QVector<double> &xDynamic,
            const QVector<double> &yDynamic,
            const QVector<double> &xGuided,
            const QVector<double> &yGuided);
    void createGraph(
            const QVector<double> &x,
            const QVector<double> &y,
            const QString &name,
            const QCPScatterStyle &style,
            const QColor &color,
            const qreal &width);
    static bool computePoint(const int type, const int sideSize, const int iters, float &avg, float &max, float &min, QProcess &process);
    void on_pushButton_clicked();

    void on_spinBox_min_valueChanged(int arg1);

    void on_spinBox_max_valueChanged(int arg1);

    void on_spinBox_step_valueChanged(int arg1);

    void closeProcess();
    QString getConvertedTime(const qint64 timeInMs);

private:
    Ui::MainWindow *ui;
    int m_minSide = 0;
    int m_maxSide = 1;
    int m_step = 1;

    QProcess m_process;
};

#endif // MAINWINDOW_H
