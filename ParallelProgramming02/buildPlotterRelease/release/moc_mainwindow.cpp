/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../Plotter/mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QVector>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[73];
    char stringdata0[801];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 16), // "titleDoubleClick"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 12), // "QMouseEvent*"
QT_MOC_LITERAL(4, 42, 5), // "event"
QT_MOC_LITERAL(5, 48, 20), // "axisLabelDoubleClick"
QT_MOC_LITERAL(6, 69, 8), // "QCPAxis*"
QT_MOC_LITERAL(7, 78, 4), // "axis"
QT_MOC_LITERAL(8, 83, 23), // "QCPAxis::SelectablePart"
QT_MOC_LITERAL(9, 107, 4), // "part"
QT_MOC_LITERAL(10, 112, 17), // "legendDoubleClick"
QT_MOC_LITERAL(11, 130, 10), // "QCPLegend*"
QT_MOC_LITERAL(12, 141, 6), // "legend"
QT_MOC_LITERAL(13, 148, 22), // "QCPAbstractLegendItem*"
QT_MOC_LITERAL(14, 171, 4), // "item"
QT_MOC_LITERAL(15, 176, 16), // "selectionChanged"
QT_MOC_LITERAL(16, 193, 10), // "mousePress"
QT_MOC_LITERAL(17, 204, 10), // "mouseWheel"
QT_MOC_LITERAL(18, 215, 14), // "addRandomGraph"
QT_MOC_LITERAL(19, 230, 19), // "removeSelectedGraph"
QT_MOC_LITERAL(20, 250, 15), // "removeAllGraphs"
QT_MOC_LITERAL(21, 266, 18), // "contextMenuRequest"
QT_MOC_LITERAL(22, 285, 3), // "pos"
QT_MOC_LITERAL(23, 289, 10), // "moveLegend"
QT_MOC_LITERAL(24, 300, 12), // "graphClicked"
QT_MOC_LITERAL(25, 313, 21), // "QCPAbstractPlottable*"
QT_MOC_LITERAL(26, 335, 9), // "plottable"
QT_MOC_LITERAL(27, 345, 9), // "dataIndex"
QT_MOC_LITERAL(28, 355, 10), // "loadGraphs"
QT_MOC_LITERAL(29, 366, 6), // "size_t"
QT_MOC_LITERAL(30, 373, 7), // "minSide"
QT_MOC_LITERAL(31, 381, 7), // "maxSide"
QT_MOC_LITERAL(32, 389, 5), // "iters"
QT_MOC_LITERAL(33, 395, 4), // "step"
QT_MOC_LITERAL(34, 400, 7), // "loadCpu"
QT_MOC_LITERAL(35, 408, 10), // "loadStatic"
QT_MOC_LITERAL(36, 419, 11), // "loadDynamic"
QT_MOC_LITERAL(37, 431, 10), // "loadGuided"
QT_MOC_LITERAL(38, 442, 12), // "updateGraphs"
QT_MOC_LITERAL(39, 455, 15), // "QVector<double>"
QT_MOC_LITERAL(40, 471, 4), // "xCpu"
QT_MOC_LITERAL(41, 476, 4), // "yCpu"
QT_MOC_LITERAL(42, 481, 7), // "xStatic"
QT_MOC_LITERAL(43, 489, 7), // "yStatic"
QT_MOC_LITERAL(44, 497, 8), // "xDynamic"
QT_MOC_LITERAL(45, 506, 8), // "yDynamic"
QT_MOC_LITERAL(46, 515, 7), // "xGuided"
QT_MOC_LITERAL(47, 523, 7), // "yGuided"
QT_MOC_LITERAL(48, 531, 11), // "createGraph"
QT_MOC_LITERAL(49, 543, 1), // "x"
QT_MOC_LITERAL(50, 545, 1), // "y"
QT_MOC_LITERAL(51, 547, 4), // "name"
QT_MOC_LITERAL(52, 552, 15), // "QCPScatterStyle"
QT_MOC_LITERAL(53, 568, 5), // "style"
QT_MOC_LITERAL(54, 574, 5), // "color"
QT_MOC_LITERAL(55, 580, 5), // "width"
QT_MOC_LITERAL(56, 586, 12), // "computePoint"
QT_MOC_LITERAL(57, 599, 4), // "type"
QT_MOC_LITERAL(58, 604, 8), // "sideSize"
QT_MOC_LITERAL(59, 613, 6), // "float&"
QT_MOC_LITERAL(60, 620, 3), // "avg"
QT_MOC_LITERAL(61, 624, 3), // "max"
QT_MOC_LITERAL(62, 628, 3), // "min"
QT_MOC_LITERAL(63, 632, 9), // "QProcess&"
QT_MOC_LITERAL(64, 642, 7), // "process"
QT_MOC_LITERAL(65, 650, 21), // "on_pushButton_clicked"
QT_MOC_LITERAL(66, 672, 27), // "on_spinBox_min_valueChanged"
QT_MOC_LITERAL(67, 700, 4), // "arg1"
QT_MOC_LITERAL(68, 705, 27), // "on_spinBox_max_valueChanged"
QT_MOC_LITERAL(69, 733, 28), // "on_spinBox_step_valueChanged"
QT_MOC_LITERAL(70, 762, 12), // "closeProcess"
QT_MOC_LITERAL(71, 775, 16), // "getConvertedTime"
QT_MOC_LITERAL(72, 792, 8) // "timeInMs"

    },
    "MainWindow\0titleDoubleClick\0\0QMouseEvent*\0"
    "event\0axisLabelDoubleClick\0QCPAxis*\0"
    "axis\0QCPAxis::SelectablePart\0part\0"
    "legendDoubleClick\0QCPLegend*\0legend\0"
    "QCPAbstractLegendItem*\0item\0"
    "selectionChanged\0mousePress\0mouseWheel\0"
    "addRandomGraph\0removeSelectedGraph\0"
    "removeAllGraphs\0contextMenuRequest\0"
    "pos\0moveLegend\0graphClicked\0"
    "QCPAbstractPlottable*\0plottable\0"
    "dataIndex\0loadGraphs\0size_t\0minSide\0"
    "maxSide\0iters\0step\0loadCpu\0loadStatic\0"
    "loadDynamic\0loadGuided\0updateGraphs\0"
    "QVector<double>\0xCpu\0yCpu\0xStatic\0"
    "yStatic\0xDynamic\0yDynamic\0xGuided\0"
    "yGuided\0createGraph\0x\0y\0name\0"
    "QCPScatterStyle\0style\0color\0width\0"
    "computePoint\0type\0sideSize\0float&\0avg\0"
    "max\0min\0QProcess&\0process\0"
    "on_pushButton_clicked\0on_spinBox_min_valueChanged\0"
    "arg1\0on_spinBox_max_valueChanged\0"
    "on_spinBox_step_valueChanged\0closeProcess\0"
    "getConvertedTime\0timeInMs"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      22,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,  124,    2, 0x08 /* Private */,
       5,    2,  127,    2, 0x08 /* Private */,
      10,    2,  132,    2, 0x08 /* Private */,
      15,    0,  137,    2, 0x08 /* Private */,
      16,    0,  138,    2, 0x08 /* Private */,
      17,    0,  139,    2, 0x08 /* Private */,
      18,    0,  140,    2, 0x08 /* Private */,
      19,    0,  141,    2, 0x08 /* Private */,
      20,    0,  142,    2, 0x08 /* Private */,
      21,    1,  143,    2, 0x08 /* Private */,
      23,    0,  146,    2, 0x08 /* Private */,
      24,    2,  147,    2, 0x08 /* Private */,
      28,    8,  152,    2, 0x08 /* Private */,
      38,    8,  169,    2, 0x08 /* Private */,
      48,    6,  186,    2, 0x08 /* Private */,
      56,    7,  199,    2, 0x08 /* Private */,
      65,    0,  214,    2, 0x08 /* Private */,
      66,    1,  215,    2, 0x08 /* Private */,
      68,    1,  218,    2, 0x08 /* Private */,
      69,    1,  221,    2, 0x08 /* Private */,
      70,    0,  224,    2, 0x08 /* Private */,
      71,    1,  225,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 8,    7,    9,
    QMetaType::Void, 0x80000000 | 11, 0x80000000 | 13,   12,   14,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QPoint,   22,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 25, QMetaType::Int,   26,   27,
    QMetaType::Void, 0x80000000 | 29, 0x80000000 | 29, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Bool, QMetaType::Bool, QMetaType::Bool,   30,   31,   32,   33,   34,   35,   36,   37,
    QMetaType::Void, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39, 0x80000000 | 39,   40,   41,   42,   43,   44,   45,   46,   47,
    QMetaType::Void, 0x80000000 | 39, 0x80000000 | 39, QMetaType::QString, 0x80000000 | 52, QMetaType::QColor, QMetaType::QReal,   49,   50,   51,   53,   54,   55,
    QMetaType::Bool, QMetaType::Int, QMetaType::Int, QMetaType::Int, 0x80000000 | 59, 0x80000000 | 59, 0x80000000 | 59, 0x80000000 | 63,   57,   58,   32,   60,   61,   62,   64,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   67,
    QMetaType::Void, QMetaType::Int,   67,
    QMetaType::Void, QMetaType::Int,   67,
    QMetaType::Void,
    QMetaType::QString, QMetaType::LongLong,   72,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->titleDoubleClick((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 1: _t->axisLabelDoubleClick((*reinterpret_cast< QCPAxis*(*)>(_a[1])),(*reinterpret_cast< QCPAxis::SelectablePart(*)>(_a[2]))); break;
        case 2: _t->legendDoubleClick((*reinterpret_cast< QCPLegend*(*)>(_a[1])),(*reinterpret_cast< QCPAbstractLegendItem*(*)>(_a[2]))); break;
        case 3: _t->selectionChanged(); break;
        case 4: _t->mousePress(); break;
        case 5: _t->mouseWheel(); break;
        case 6: _t->addRandomGraph(); break;
        case 7: _t->removeSelectedGraph(); break;
        case 8: _t->removeAllGraphs(); break;
        case 9: _t->contextMenuRequest((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        case 10: _t->moveLegend(); break;
        case 11: _t->graphClicked((*reinterpret_cast< QCPAbstractPlottable*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 12: _t->loadGraphs((*reinterpret_cast< const size_t(*)>(_a[1])),(*reinterpret_cast< const size_t(*)>(_a[2])),(*reinterpret_cast< const int(*)>(_a[3])),(*reinterpret_cast< const int(*)>(_a[4])),(*reinterpret_cast< const bool(*)>(_a[5])),(*reinterpret_cast< const bool(*)>(_a[6])),(*reinterpret_cast< const bool(*)>(_a[7])),(*reinterpret_cast< const bool(*)>(_a[8]))); break;
        case 13: _t->updateGraphs((*reinterpret_cast< const QVector<double>(*)>(_a[1])),(*reinterpret_cast< const QVector<double>(*)>(_a[2])),(*reinterpret_cast< const QVector<double>(*)>(_a[3])),(*reinterpret_cast< const QVector<double>(*)>(_a[4])),(*reinterpret_cast< const QVector<double>(*)>(_a[5])),(*reinterpret_cast< const QVector<double>(*)>(_a[6])),(*reinterpret_cast< const QVector<double>(*)>(_a[7])),(*reinterpret_cast< const QVector<double>(*)>(_a[8]))); break;
        case 14: _t->createGraph((*reinterpret_cast< const QVector<double>(*)>(_a[1])),(*reinterpret_cast< const QVector<double>(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3])),(*reinterpret_cast< const QCPScatterStyle(*)>(_a[4])),(*reinterpret_cast< const QColor(*)>(_a[5])),(*reinterpret_cast< const qreal(*)>(_a[6]))); break;
        case 15: { bool _r = _t->computePoint((*reinterpret_cast< const int(*)>(_a[1])),(*reinterpret_cast< const int(*)>(_a[2])),(*reinterpret_cast< const int(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4])),(*reinterpret_cast< float(*)>(_a[5])),(*reinterpret_cast< float(*)>(_a[6])),(*reinterpret_cast< QProcess(*)>(_a[7])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 16: _t->on_pushButton_clicked(); break;
        case 17: _t->on_spinBox_min_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 18: _t->on_spinBox_max_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: _t->on_spinBox_step_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: _t->closeProcess(); break;
        case 21: { QString _r = _t->getConvertedTime((*reinterpret_cast< const qint64(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPAxis* >(); break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPAxis::SelectablePart >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPAbstractLegendItem* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPLegend* >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCPAbstractPlottable* >(); break;
            }
            break;
        case 13:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 7:
            case 6:
            case 5:
            case 4:
            case 3:
            case 2:
            case 1:
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVector<double> >(); break;
            }
            break;
        case 14:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVector<double> >(); break;
            }
            break;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.data,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 22)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 22;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 22)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 22;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
