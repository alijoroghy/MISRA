import sys, json, pathlib, traceback
from dataclasses import asdict
import numpy as np
from engine import Config, trace
from PySide6.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
 QDoubleSpinBox,QSpinBox,QComboBox,QPushButton,QLabel,QFileDialog,QMessageBox,QTabWidget,QGroupBox,QScrollArea,QSplashScreen)
from PySide6.QtGui import QPixmap,QPainter,QColor,QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Plot(FigureCanvasQTAgg):
    def __init__(self):
        self.fig=Figure(figsize=(8,5),layout='constrained')
        super().__init__(self.fig)
    def render(self,data):
        self.fig.clear()
        ax=self.fig.subplots(2,3)
        arr=data['frames']; mask=data['valid']; opd=data['opd_m']*1e9
        views=[(arr[0],'Detector frame 0', 'gray'),(arr[min(1,len(arr)-1)],'Detector frame 1','gray'),
               (np.where(mask,opd,np.nan),'Ground-truth OPD (nm)','RdBu_r'),
               (np.where(mask,data['phase_rad'],np.nan),'Wrapped optical phase (rad)','twilight'),
               (data['reference']['segment_lengths'][1]*1e3,'Reference hit distance (mm)','viridis'),
               (data['test']['segment_lengths'][1]*1e3,'Test hit distance (mm)','viridis')]
        for a,(im,title,cmap) in zip(ax.flat,views):
            a.imshow(im,cmap=cmap,origin='lower');a.set_title(title,fontsize=9);a.set_xticks([]);a.set_yticks([])
        self.draw_idle()

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('MISRA — Michelson Interferometry Simulator by Ali | v0.6')
        self.resize(1350,820);self.data=None;self.cfg=None
        root=QWidget();self.setCentralWidget(root);outer=QHBoxLayout(root)
        sc=QScrollArea();sc.setWidgetResizable(True);sc.setFixedWidth(345)
        panel=QWidget();formbox=QVBoxLayout(panel)
        logo=QLabel('MISRA');logo.setStyleSheet('font-size:30px;font-weight:bold;color:#4ea5f5');formbox.addWidget(logo)
        formbox.addWidget(QLabel('Michelson Interferometry Simulator\nby Ali'))
        group=QGroupBox('Physical parameters');form=QFormLayout(group);formbox.addWidget(group)
        self.controls={}
        def num(key,label,value,low,high,dec=4,integer=False):
            s=QSpinBox() if integer else QDoubleSpinBox();s.setRange(low,high)
            if not integer:s.setDecimals(dec)
            s.setValue(value);form.addRow(label,s);self.controls[key]=s
        def choice(key,label,options,current):
            s=QComboBox();s.addItems(options);s.setCurrentText(current);form.addRow(label,s);self.controls[key]=s
        defaults=Config()
        choice('mirror','Test mirror',['flat','spherical','parabolic'],defaults.mirror)
        num('radius_mm','Signed curvature R (mm)',defaults.radius_mm,-100000,100000,3)
        num('tilt_y_mrad','Tilt Y (mrad)',0,-100,100)
        num('tilt_z_mrad','Tilt Z (mrad)',0,-100,100)
        num('reference_mm','Reference arm (mm)',50,1,1000)
        num('test_mm','Test arm (mm)',50,1,1000)
        num('detector_mm','Detector distance (mm)',50,1,1000)
        num('source_mm','Source plane distance (mm)',70,1,1000)
        choice('source','Illumination',['plane','point'],'plane')
        num('wavelength_nm','Wavelength (nm)',632.8,200,2000)
        num('aperture_mm','Mirror aperture (mm)',8,0.1,100)
        num('fov_mm','Detector FOV (mm)',3,0.1,20)
        num('n','Pixels per side',256,16,1024,integer=True)
        choice('phase_steps','Detector captures',['1','4'],'4')
        num('noise_std','Detector Gaussian noise',0,0,0.5,5)
        num('seed','Noise seed',42,0,999999,integer=True)
        run=QPushButton('Trace rays + simulate detector');run.clicked.connect(self.simulate);formbox.addWidget(run)
        save=QPushButton('Export detector dataset');save.clicked.connect(self.export);formbox.addWidget(save)
        formbox.addWidget(QLabel('Output is RAW detector intensity.\nNo surface reconstruction is performed.'))
        formbox.addStretch();sc.setWidget(panel);outer.addWidget(sc)
        tabs=QTabWidget();outer.addWidget(tabs,1)
        self.plot=Plot();tabs.addTab(self.plot,'Detector & optical paths')
        self.setup=Plot();tabs.addTab(self.setup,'Optical setup schematic')
        self.statusBar().showMessage('Ready')
        self.simulate()
    def config(self):
        kw={k:(int(w.currentText()) if k=='phase_steps' else w.currentText()) if isinstance(w,QComboBox) else w.value() for k,w in self.controls.items()}
        return Config(**kw)
    def simulate(self):
        try:
            cfg=self.config();d=trace(cfg);self.data=d;self.cfg=cfg
            self.plot.render(d);self.draw_setup()
            self.statusBar().showMessage(f"{cfg.n**2:,} rays per arm; valid {d['valid'].mean()*100:.1f}% | {len(d['frames'])} frame(s)")
        except Exception as e: QMessageBox.critical(self,'Simulation error',str(e)+'\n\n'+traceback.format_exc(limit=3))
    def draw_setup(self):
        fig=self.setup.fig;fig.clear();ax=fig.add_subplot(111,projection='3d')
        c=self.cfg;d=self.data
        bs=np.array([[0,0,0],[.008,.008,0]])
        ax.plot(bs[:,0]*1e3,bs[:,1]*1e3,bs[:,2]*1e3,color='gray',lw=6,label='BS (x=y)')
        ax.scatter([0,c.test_mm,-c.source_mm],[0,0,0],[0,0,0],c=['black','orange','blue'])
        ax.scatter([0],[c.reference_mm],[0],c='green')
        ax.scatter([0],[-c.detector_mm],[0],c='red')
        for arm,col in [('reference','green'),('test','orange')]:
            m=d[arm]['mirror_xyz'];b=d[arm]['return_bs_xyz'];s=d[arm]['source_xyz']
            ids=[(c.n//2,c.n//2),(c.n//3,c.n//3),(2*c.n//3,2*c.n//3)]
            for i,j in ids:
                if not d[arm]['valid'][i,j]:continue
                a=np.array([d['x_m'][j],-c.detector_mm*1e-3,d['z_m'][i]])
                bb=np.array([a[0],a[0],a[2]])
                points=np.stack([a,bb,m[i,j],b[i,j],s[i,j]])*1e3
                ax.plot(points[:,0],points[:,1],points[:,2],color=col,alpha=.8)
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_zlabel('z (mm)')
        ax.set_title('Reverse-traced rays: detector → mirrors → source plane')
        self.setup.draw_idle()
    def export(self):
        if self.data is None:return
        folder=QFileDialog.getExistingDirectory(self,'Choose output directory')
        if not folder:return
        path=pathlib.Path(folder);d=self.data;c=self.cfg
        # measurement files intentionally contain NO truth OPD, phase or surface profile
        for i,frame in enumerate(d['frames']):
            np.savetxt(path/f'detector_{i:02d}.csv',frame,delimiter=',',fmt='%.8f')
        np.savez_compressed(path/'detector_dataset.npz',intensity=d['frames'],
                            x_m=d['x_m'],z_m=d['z_m'],phase_shifts_rad=d['phase_shifts_rad'],
                            valid_mask=d['valid'])
        try:
            from PIL import Image
            for i,frame in enumerate(d['frames']):Image.fromarray(np.uint16(np.clip(frame,0,1)*65535)).save(path/f'detector_{i:02d}.png')
        except ImportError:pass
        meta=asdict(c);meta.update({'simulator':'MISRA v0.6 by Ali','units':'SI arrays in NPZ; intensity normalized 0..1',
                                   'note':'Phase and ground-truth OPD excluded from detector export. Geometric-optics approximation.',
                                   'phase_shifts_rad':d['phase_shifts_rad'].tolist()})
        (path/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        # ground truth isolated for benchmark, never part of measurement file
        np.savez_compressed(path/'SIMULATION_TRUTH_DO_NOT_FEED_RECONSTRUCTION.npz',
                            opd_m=d['opd_m'],phase_rad=d['phase_rad'],valid=d['valid'],
                            reference_hit_m=d['reference']['mirror_xyz'],test_hit_m=d['test']['mirror_xyz'])
        QMessageBox.information(self,'Export completed',f'Files saved to:\n{path}')

def main():
    app=QApplication(sys.argv)
    pm=QPixmap(640,300);pm.fill(QColor('#101c32'))
    painter=QPainter(pm);painter.setPen(QColor('#cbe8ff'));font=QFont('Arial',55,QFont.Weight.Bold);painter.setFont(font)
    painter.drawText(QRect(0,50,640,100),Qt.AlignmentFlag.AlignCenter,'MISRA')
    painter.setFont(QFont('Arial',17));painter.drawText(QRect(0,145,640,45),Qt.AlignmentFlag.AlignCenter,'Michelson Interferometry Simulator')
    painter.drawText(QRect(0,195,640,40),Qt.AlignmentFlag.AlignCenter,'by Ali');painter.end()
    splash=QSplashScreen(pm);splash.show();app.processEvents()
    def start():
        w=Window();w.show();splash.finish(w);app.window=w
    QTimer.singleShot(2000,start)
    sys.exit(app.exec())
if __name__=='__main__':main()
