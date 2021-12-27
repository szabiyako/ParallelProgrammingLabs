#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QElapsedTimer>

const size_t maxElements = 19600000000; //2147395600
const size_t maxTime = 200;
const QString cmdFilePath = "../x64/Release/ParallelProgramming02.exe";

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    std::srand(QDateTime::currentDateTime().toMSecsSinceEpoch()/1000.0);
    ui->setupUi(this);

    qDebug() << "OpenGL:" << ui->customPlot->openGl();
    ui->customPlot->setOpenGl(true, 4);
    qDebug() << "OpenGL:" << ui->customPlot->openGl();

    ui->customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes |
                                    QCP::iSelectLegend | QCP::iSelectPlottables);
    ui->customPlot->xAxis->setRange(0, maxElements);
    ui->customPlot->yAxis->setRange(0, maxTime);
    ui->customPlot->axisRect()->setupFullAxesBox();

    ui->customPlot->plotLayout()->insertRow(0);
    QCPTextElement *title = new QCPTextElement(ui->customPlot, "Performance", QFont("sans", 17, QFont::Bold));
    ui->customPlot->plotLayout()->addElement(0, 0, title);

    ui->customPlot->xAxis->setLabel("Elements in matrix");
    ui->customPlot->yAxis->setLabel("Time ms");
    ui->customPlot->legend->setVisible(true);
    QFont legendFont = font();
    legendFont.setPointSize(10);
    ui->customPlot->legend->setFont(legendFont);
    ui->customPlot->legend->setSelectedFont(legendFont);
    ui->customPlot->legend->setSelectableParts(QCPLegend::spItems); // legend box shall not be selectable, only legend items

    //addRandomGraph();
    //addRandomGraph();
    //addRandomGraph();
    //addRandomGraph();
    //ui->customPlot->rescaleAxes();

    // connect slot that ties some axis selections together (especially opposite axes):
    connect(ui->customPlot, SIGNAL(selectionChangedByUser()), this, SLOT(selectionChanged()));
    // connect slots that takes care that when an axis is selected, only that direction can be dragged and zoomed:
    connect(ui->customPlot, SIGNAL(mousePress(QMouseEvent*)), this, SLOT(mousePress()));
    connect(ui->customPlot, SIGNAL(mouseWheel(QWheelEvent*)), this, SLOT(mouseWheel()));

    // make bottom and left axes transfer their ranges to top and right axes:
    connect(ui->customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->yAxis2, SLOT(setRange(QCPRange)));

    // connect some interaction slots:
    connect(ui->customPlot, SIGNAL(axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(axisLabelDoubleClick(QCPAxis*,QCPAxis::SelectablePart)));
    connect(ui->customPlot, SIGNAL(legendDoubleClick(QCPLegend*,QCPAbstractLegendItem*,QMouseEvent*)), this, SLOT(legendDoubleClick(QCPLegend*,QCPAbstractLegendItem*)));
    connect(title, SIGNAL(doubleClicked(QMouseEvent*)), this, SLOT(titleDoubleClick(QMouseEvent*)));

    // connect slot that shows a message in the status bar when a graph is clicked:
    connect(ui->customPlot, SIGNAL(plottableClick(QCPAbstractPlottable*,int,QMouseEvent*)), this, SLOT(graphClicked(QCPAbstractPlottable*,int)));

    // setup policy and connect slot for context menu popup:
    ui->customPlot->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->customPlot, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(contextMenuRequest(QPoint)));
}

MainWindow::~MainWindow()
{
    delete ui;

    closeProcess();
    qDebug() << "CLOSED";
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    closeProcess();
    event->accept();
    qDebug() << "Close event";
}

void MainWindow::titleDoubleClick(QMouseEvent* event)
{
    Q_UNUSED(event)
    if (QCPTextElement *title = qobject_cast<QCPTextElement*>(sender()))
    {
        // Set the plot title by double clicking on it
        bool ok;
        QString newTitle = QInputDialog::getText(this, "QCustomPlot example", "New plot title:", QLineEdit::Normal, title->text(), &ok);
        if (ok)
        {
            title->setText(newTitle);
            ui->customPlot->replot();
        }
    }
}

void MainWindow::axisLabelDoubleClick(QCPAxis *axis, QCPAxis::SelectablePart part)
{
    // Set an axis label by double clicking on it
    if (part == QCPAxis::spAxisLabel) // only react when the actual axis label is clicked, not tick label or axis backbone
    {
        bool ok;
        QString newLabel = QInputDialog::getText(this, "QCustomPlot example", "New axis label:", QLineEdit::Normal, axis->label(), &ok);
        if (ok)
        {
            axis->setLabel(newLabel);
            ui->customPlot->replot();
        }
    }
}

void MainWindow::legendDoubleClick(QCPLegend *legend, QCPAbstractLegendItem *item)
{
    // Rename a graph by double clicking on its legend item
    Q_UNUSED(legend)
    if (item) // only react if item was clicked (user could have clicked on border padding of legend where there is no item, then item is 0)
    {
        QCPPlottableLegendItem *plItem = qobject_cast<QCPPlottableLegendItem*>(item);
        bool ok;
        QString newName = QInputDialog::getText(this, "QCustomPlot example", "New graph name:", QLineEdit::Normal, plItem->plottable()->name(), &ok);
        if (ok)
        {
            plItem->plottable()->setName(newName);
            ui->customPlot->replot();
        }
    }
}

void MainWindow::selectionChanged()
{
    /*
   normally, axis base line, axis tick labels and axis labels are selectable separately, but we want
   the user only to be able to select the axis as a whole, so we tie the selected states of the tick labels
   and the axis base line together. However, the axis label shall be selectable individually.
   
   The selection state of the left and right axes shall be synchronized as well as the state of the
   bottom and top axes.
   
   Further, we want to synchronize the selection of the graphs with the selection state of the respective
   legend item belonging to that graph. So the user can select a graph by either clicking on the graph itself
   or on its legend item.
  */

    // make top and bottom axes be selected synchronously, and handle axis and tick labels as one selectable object:
    if (ui->customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis) || ui->customPlot->xAxis->selectedParts().testFlag(QCPAxis::spTickLabels) ||
            ui->customPlot->xAxis2->selectedParts().testFlag(QCPAxis::spAxis) || ui->customPlot->xAxis2->selectedParts().testFlag(QCPAxis::spTickLabels))
    {
        ui->customPlot->xAxis2->setSelectedParts(QCPAxis::spAxis|QCPAxis::spTickLabels);
        ui->customPlot->xAxis->setSelectedParts(QCPAxis::spAxis|QCPAxis::spTickLabels);
    }
    // make left and right axes be selected synchronously, and handle axis and tick labels as one selectable object:
    if (ui->customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis) || ui->customPlot->yAxis->selectedParts().testFlag(QCPAxis::spTickLabels) ||
            ui->customPlot->yAxis2->selectedParts().testFlag(QCPAxis::spAxis) || ui->customPlot->yAxis2->selectedParts().testFlag(QCPAxis::spTickLabels))
    {
        ui->customPlot->yAxis2->setSelectedParts(QCPAxis::spAxis|QCPAxis::spTickLabels);
        ui->customPlot->yAxis->setSelectedParts(QCPAxis::spAxis|QCPAxis::spTickLabels);
    }

    // synchronize selection of graphs with selection of corresponding legend items:
    for (int i=0; i<ui->customPlot->graphCount(); ++i)
    {
        QCPGraph *graph = ui->customPlot->graph(i);
        QCPPlottableLegendItem *item = ui->customPlot->legend->itemWithPlottable(graph);
        if (item->selected() || graph->selected())
        {
            item->setSelected(true);
            graph->setSelection(QCPDataSelection(graph->data()->dataRange()));
        }
    }
}

void MainWindow::mousePress()
{
    // if an axis is selected, only allow the direction of that axis to be dragged
    // if no axis is selected, both directions may be dragged

    if (ui->customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
        ui->customPlot->axisRect()->setRangeDrag(ui->customPlot->xAxis->orientation());
    else if (ui->customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
        ui->customPlot->axisRect()->setRangeDrag(ui->customPlot->yAxis->orientation());
    else
        ui->customPlot->axisRect()->setRangeDrag(Qt::Horizontal|Qt::Vertical);
}

void MainWindow::mouseWheel()
{
    // if an axis is selected, only allow the direction of that axis to be zoomed
    // if no axis is selected, both directions may be zoomed

    if (ui->customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
        ui->customPlot->axisRect()->setRangeZoom(ui->customPlot->xAxis->orientation());
    else if (ui->customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
        ui->customPlot->axisRect()->setRangeZoom(ui->customPlot->yAxis->orientation());
    else
        ui->customPlot->axisRect()->setRangeZoom(Qt::Horizontal|Qt::Vertical);
}

void MainWindow::addRandomGraph()
{
    int n = 50; // number of points in graph
    double xScale = (std::rand()/(double)RAND_MAX + 0.5)*2;
    double yScale = (std::rand()/(double)RAND_MAX + 0.5)*2;
    double xOffset = (std::rand()/(double)RAND_MAX - 0.5)*4;
    double yOffset = (std::rand()/(double)RAND_MAX - 0.5)*10;
    double r1 = (std::rand()/(double)RAND_MAX - 0.5)*2;
    double r2 = (std::rand()/(double)RAND_MAX - 0.5)*2;
    double r3 = (std::rand()/(double)RAND_MAX - 0.5)*2;
    double r4 = (std::rand()/(double)RAND_MAX - 0.5)*2;
    QVector<double> x(n), y(n);
    for (int i=0; i<n; i++)
    {
        x[i] = (i/(double)n-0.5)*10.0*xScale + xOffset;
        y[i] = (qSin(x[i]*r1*5)*qSin(qCos(x[i]*r2)*r4*3)+r3*qCos(qSin(x[i])*r4*2))*yScale + yOffset;
    }

    ui->customPlot->addGraph();
    ui->customPlot->graph()->setName(QString("New graph %1").arg(ui->customPlot->graphCount()-1));
    ui->customPlot->graph()->setData(x, y);
    ui->customPlot->graph()->setLineStyle((QCPGraph::LineStyle)(std::rand()%5+1));
    if (std::rand()%100 > 50)
        ui->customPlot->graph()->setScatterStyle(QCPScatterStyle((QCPScatterStyle::ScatterShape)(std::rand()%14+1)));
    QPen graphPen;
    graphPen.setColor(QColor(std::rand()%245+10, std::rand()%245+10, std::rand()%245+10));
    graphPen.setWidthF(std::rand()/(double)RAND_MAX*2+1);
    ui->customPlot->graph()->setPen(graphPen);
    ui->customPlot->replot();
}

void MainWindow::removeSelectedGraph()
{
    if (ui->customPlot->selectedGraphs().size() > 0)
    {
        ui->customPlot->removeGraph(ui->customPlot->selectedGraphs().first());
        ui->customPlot->replot();
    }
}

void MainWindow::removeAllGraphs()
{
    ui->customPlot->clearGraphs();
    ui->customPlot->replot();
}

void MainWindow::contextMenuRequest(QPoint pos)
{
    QMenu *menu = new QMenu(this);
    menu->setAttribute(Qt::WA_DeleteOnClose);

    if (ui->customPlot->legend->selectTest(pos, false) >= 0) // context menu on legend requested
    {
        menu->addAction("Move to top left", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop|Qt::AlignLeft));
        menu->addAction("Move to top center", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop|Qt::AlignHCenter));
        menu->addAction("Move to top right", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop|Qt::AlignRight));
        menu->addAction("Move to bottom right", this, SLOT(moveLegend()))->setData((int)(Qt::AlignBottom|Qt::AlignRight));
        menu->addAction("Move to bottom left", this, SLOT(moveLegend()))->setData((int)(Qt::AlignBottom|Qt::AlignLeft));
    } else  // general context menu on graphs requested
    {
        menu->addAction("Add random graph", this, SLOT(addRandomGraph()));
        if (ui->customPlot->selectedGraphs().size() > 0)
            menu->addAction("Remove selected graph", this, SLOT(removeSelectedGraph()));
        if (ui->customPlot->graphCount() > 0)
            menu->addAction("Remove all graphs", this, SLOT(removeAllGraphs()));
    }

    menu->popup(ui->customPlot->mapToGlobal(pos));
}

void MainWindow::moveLegend()
{
    if (QAction* contextAction = qobject_cast<QAction*>(sender())) // make sure this slot is really called by a context menu action, so it carries the data we need
    {
        bool ok;
        int dataInt = contextAction->data().toInt(&ok);
        if (ok)
        {
            ui->customPlot->axisRect()->insetLayout()->setInsetAlignment(0, (Qt::Alignment)dataInt);
            ui->customPlot->replot();
        }
    }
}

void MainWindow::graphClicked(QCPAbstractPlottable *plottable, int dataIndex)
{
    // since we know we only have QCPGraphs in the plot, we can immediately access interface1D()
    // usually it's better to first check whether interface1D() returns non-zero, and only then use it.
    double dataValue = plottable->interface1D()->dataMainValue(dataIndex);
    QString message = QString("Clicked on graph '%1' at data point #%2 with value %3.").arg(plottable->name()).arg(dataIndex).arg(dataValue);
    ui->statusBar->showMessage(message, 2500);
}

void MainWindow::loadGraphs(const size_t minSide,
                            const size_t maxSide,
                            const int iters,
                            const int step,
                            const bool loadCpu,
                            const bool loadStatic,
                            const bool loadDynamic,
                            const bool loadGuided)
{
    const int nTotalValues = maxSide - minSide + 1;

    const int nValues = nTotalValues / step;

    QVector<double> xCpu;
    QVector<double> yCpu;
    QVector<double> xStatic;
    QVector<double> yStatic;
    QVector<double> xDynamic;
    QVector<double> yDynamic;
    QVector<double> xGuided;
    QVector<double> yGuided;

    if (loadCpu) {
        xCpu.reserve(nValues);
        yCpu.reserve(nValues);
    }
    if (loadStatic) {
        xStatic.reserve(nValues);
        yStatic.reserve(nValues);
    }
    if (loadDynamic) {
        xDynamic.reserve(nValues);
        yDynamic.reserve(nValues);
    }
    if (loadGuided) {
        xGuided.reserve(nValues);
        yGuided.reserve(nValues);
    }

    m_process.setCreateProcessArgumentsModifier([] (QProcess::CreateProcessArguments *args)
    {
        //args->flags |= CREATE_NEW_CONSOLE | REALTIME_PRIORITY_CLASS;
        args->flags |= REALTIME_PRIORITY_CLASS;
        //args->startupInfo->dwFlags &= ~STARTF_USESTDHANDLES;
        //args->startupInfo->dwFlags |= STARTF_USEFILLATTRIBUTE;
        ////
        //args->startupInfo->dwFlags |= STARTF_USESHOWWINDOW;
        //args->startupInfo->wShowWindow = SW_SHOWMINNOACTIVE;
        ////
        //args->startupInfo->dwFillAttribute = FOREGROUND_GREEN;
    });
    m_process.start(
                cmdFilePath,
                QStringList() << QString::number(maxSide));

    m_process.waitForStarted(30000);

    for (int i = 0; i < nValues; ++i) {
        int sideSize = minSide + (i * step);
        const float x = float(sideSize) * float(sideSize);
        float avg;
        float max;
        float min;

        if (loadCpu) {
            computePoint(0, sideSize, iters, avg, max, min, m_process);
            xCpu.push_back(x);
            yCpu.push_back(avg);
        }
        if (loadStatic) {
            computePoint(1, sideSize, iters, avg, max, min, m_process);
            xStatic.push_back(x);
            yStatic.push_back(avg);
        }
        if (loadDynamic) {
            computePoint(2, sideSize, iters, avg, max, min, m_process);
            xDynamic.push_back(x);
            yDynamic.push_back(avg);
        }
        if (loadGuided) {
            computePoint(3, sideSize, iters, avg, max, min, m_process);
            xGuided.push_back(x);
            yGuided.push_back(avg);
        }

        updateGraphs(xCpu,
                     yCpu,
                     xStatic,
                     yStatic,
                     xDynamic,
                     yDynamic,
                     xGuided,
                     yGuided);
    }
    m_process.write("-1\n");
}

void MainWindow::updateGraphs(const QVector<double> &xCpu,
                              const QVector<double> &yCpu,
                              const QVector<double> &xStatic,
                              const QVector<double> &yStatic,
                              const QVector<double> &xDynamic,
                              const QVector<double> &yDynamic,
                              const QVector<double> &xGuided,
                              const QVector<double> &yGuided)
{
    ui->customPlot->clearGraphs();

    createGraph(xCpu, yCpu, "Single thread", QCPScatterStyle::ScatterShape::ssNone, QColor(255, 0, 0), 3);
    createGraph(xStatic, yStatic, "OpenCV Static", QCPScatterStyle::ScatterShape::ssNone, QColor(0, 0, 255), 3);
    createGraph(xDynamic, yDynamic, "OpenCV Dynamic", QCPScatterStyle::ScatterShape::ssNone, QColor(255, 255, 0), 3);
    createGraph(xGuided, yGuided, "OpenCV Guided", QCPScatterStyle::ScatterShape::ssNone, QColor(0, 255, 0), 3);
    ui->customPlot->rescaleAxes();

    ui->customPlot->replot();
}

void MainWindow::createGraph(const QVector<double> &x,
                             const QVector<double> &y,
                             const QString &name,
                             const QCPScatterStyle &style,
                             const QColor &color,
                             const qreal &width)
{
    ui->customPlot->addGraph();
    ui->customPlot->graph()->setName(name);
    ui->customPlot->graph()->setData(x, y, true);
    ui->customPlot->graph()->setLineStyle(QCPGraph::LineStyle::lsLine);
    ui->customPlot->graph()->setScatterStyle(style);
    QPen graphPen;
    graphPen.setColor(color);
    graphPen.setWidthF(width);
    ui->customPlot->graph()->setPen(graphPen);
}

#include <iostream>

bool MainWindow::computePoint(const int type, const int sideSize, const int iters, float &avg, float &max, float &min, QProcess &process)
{
    std::string params = std::to_string(type) + " " + std::to_string(sideSize) + " " + std::to_string(iters) + "\n";

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Bytes writed = " << process.write(params.data(), params.length()) << std::endl;
    std::cout << params.data() << std::endl;

    process.waitForBytesWritten(-1);
    //process.errorString()

    bool done = false;
    while (!done) {
        if (process.waitForReadyRead(-1))
            //if (process.waitForReadyRead(10))
            //if (process.canReadLine())
                done = true;
        QApplication::processEvents(); // TODO REMOVE
    }
    const size_t bufferSize = 256;
    char buffer[256];
    std::cout << "Bytes read = " << process.readLine(buffer, bufferSize) << std::endl;
    std::cout << buffer << std::endl;
    std::cout << "--------------------------------------" << std::endl << std::endl << std::endl;


    const QString output(buffer);
    const QStringList outputValues = output.split(' ', Qt::SplitBehaviorFlags::SkipEmptyParts);
    avg = outputValues[0].toFloat() / 1000000.f;
    max = outputValues[1].toFloat() / 1000000.f;
    min = outputValues[2].toFloat() / 1000000.f;
    return true;
}


void MainWindow::on_pushButton_clicked()
{
    QElapsedTimer timer;
    timer.start();
    loadGraphs(ui->spinBox_min->value(), ui->spinBox_max->value(), ui->spinBox_iters->value(), ui->spinBox_step->value(), true, true, true, true);
    qint64 elapsedTimeMs =  timer.elapsed();

    QMessageBox::information(this, "Done", "Plotting finished in  " + getConvertedTime(elapsedTimeMs));
}

void MainWindow::on_spinBox_min_valueChanged(int arg1)
{
    if (arg1 > ui->spinBox_max->value())
        ui->spinBox_min->setValue(m_minSide);
    else
        m_minSide = arg1;
    ui->label_firstElems->setText("(" + QString::number(float(size_t(m_minSide) * size_t(m_minSide))) + ")");
}

void MainWindow::on_spinBox_max_valueChanged(int arg1)
{
    if (arg1 < ui->spinBox_min->value())
        ui->spinBox_max->setValue(m_maxSide);
    else
        m_maxSide = arg1;
    ui->label_lastElems->setText("(" + QString::number(float(size_t(m_maxSide) * size_t(m_maxSide))) + ")");
}

void MainWindow::on_spinBox_step_valueChanged(int arg1)
{
    //if (arg1 < )
}

void MainWindow::closeProcess()
{
    if (m_process.state() != QProcess::ProcessState::NotRunning) {
        qWarning().nospace()
            << "QProcess: Destroyed while process (" << QDir::toNativeSeparators(m_process.program()) << ") is still running.";
        m_process.kill();
        m_process.waitForFinished(-1);
    }
}

QString MainWindow::getConvertedTime(const qint64 timeInMs)
{
    const qint64 timeInSeconds = timeInMs / 1000;
    const qint64 timeInMinutes = timeInSeconds / 60;
    const qint64 timeInHours = timeInMinutes / 60;
    const qint64 ms = timeInMs % 1000;
    const qint64 sec = timeInSeconds % 60;
    const qint64 min = timeInMinutes % 60;
    const qint64 hrs = timeInHours;

    QString result;
    if (hrs > 0)
        result += QString::number(hrs) + "h, ";
    if (min > 0)
        result += QString::number(min) + "min, ";
    if (sec > 0)
        result += QString::number(sec) + "sec, ";
    result += QString::number(ms) + "ms";

    return result;
}
