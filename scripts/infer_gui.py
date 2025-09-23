import os
import threading
import queue
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch

# Ensure repository root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.infer as infer_mod


class InferenceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Pathothai Inference')
        self.geometry('720x520')
        try:
            self.iconbitmap(False)
        except Exception:
            pass

        self._cancel_event = threading.Event()
        self._thread = None
        self._log_queue = queue.Queue()

        self._build_ui()

    def _build_ui(self):
    pad = {'padx': 8, 'pady': 6}

    frm = ttk.Frame(self)
    frm.pack(fill='both', expand=True)

    # Input path
    ttk.Label(frm, text='Input (slide or folder):').grid(row=0, column=0, sticky='w', **pad)
    self.input_var = tk.StringVar(value='data')
    ttk.Entry(frm, textvariable=self.input_var, width=60).grid(row=0, column=1, sticky='we', **pad)
    ttk.Button(frm, text='Browse…', command=self._browse_input).grid(row=0, column=2, **pad)

    # Checkpoint
    ttk.Label(frm, text='Checkpoint:').grid(row=1, column=0, sticky='w', **pad)
    self.ckpt_var = tk.StringVar(value=str(Path('checkpoints') / 'best_model.pth'))
    ttk.Entry(frm, textvariable=self.ckpt_var, width=60).grid(row=1, column=1, sticky='we', **pad)
    ttk.Button(frm, text='Browse…', command=self._browse_ckpt).grid(row=1, column=2, **pad)

    # Annotations dir
    ttk.Label(frm, text='Annotations dir (optional):').grid(row=2, column=0, sticky='w', **pad)
    self.ann_var = tk.StringVar(value='Annotations')
    ttk.Entry(frm, textvariable=self.ann_var, width=60).grid(row=2, column=1, sticky='we', **pad)
    ttk.Button(frm, text='Browse…', command=self._browse_ann).grid(row=2, column=2, **pad)

    # Output JSON
    ttk.Label(frm, text='Output JSON:').grid(row=3, column=0, sticky='w', **pad)
    self.out_var = tk.StringVar(value=str(Path('output') / 'inference_results.json'))
    ttk.Entry(frm, textvariable=self.out_var, width=60).grid(row=3, column=1, sticky='we', **pad)
    ttk.Button(frm, text='Browse…', command=self._browse_out).grid(row=3, column=2, **pad)

    # Optional ROI classifier (for segmentation)
    ttk.Label(frm, text='ROI classifier ckpt (optional):').grid(row=4, column=0, sticky='w', **pad)
    self.ckpt_cls_var = tk.StringVar(value='')
    ttk.Entry(frm, textvariable=self.ckpt_cls_var, width=60).grid(row=4, column=1, sticky='we', **pad)
    ttk.Button(frm, text='Browse…', command=self._browse_ckpt_cls).grid(row=4, column=2, **pad)

    # Params frame
    pfrm = ttk.LabelFrame(frm, text='Parameters')
    pfrm.grid(row=5, column=0, columnspan=3, sticky='we', **pad)
    pfrm.columnconfigure(5, weight=1)

    ttk.Label(pfrm, text='Batch size').grid(row=0, column=0, sticky='w', **pad)
    self.bs_var = tk.IntVar(value=16)
    ttk.Spinbox(pfrm, from_=1, to=1024, textvariable=self.bs_var, width=8).grid(row=0, column=1, **pad)

    ttk.Label(pfrm, text='Patch size').grid(row=0, column=2, sticky='w', **pad)
    self.ps_var = tk.IntVar(value=224)
    ttk.Spinbox(pfrm, from_=64, to=2048, increment=32, textvariable=self.ps_var, width=8).grid(row=0, column=3, **pad)

    ttk.Label(pfrm, text='Patches/slide').grid(row=0, column=4, sticky='w', **pad)
    self.pps_var = tk.IntVar(value=100)
    ttk.Spinbox(pfrm, from_=1, to=10000, textvariable=self.pps_var, width=8).grid(row=0, column=5, **pad)

    ttk.Label(pfrm, text='SVS level').grid(row=1, column=0, sticky='w', **pad)
    self.level_var = tk.IntVar(value=1)
    ttk.Spinbox(pfrm, from_=0, to=8, textvariable=self.level_var, width=8).grid(row=1, column=1, **pad)

    ttk.Label(pfrm, text='Num workers').grid(row=1, column=2, sticky='w', **pad)
    self.nw_var = tk.IntVar(value=0)
    ttk.Spinbox(pfrm, from_=0, to=32, textvariable=self.nw_var, width=8).grid(row=1, column=3, **pad)

    ttk.Label(pfrm, text='Device').grid(row=1, column=4, sticky='w', **pad)
    self.dev_var = tk.StringVar(value='cuda' if torch.cuda.is_available() else 'cpu')
    ttk.Combobox(pfrm, textvariable=self.dev_var, values=['cuda', 'cpu'], state='readonly', width=8).grid(row=1, column=5, **pad)

    ttk.Label(pfrm, text='Task').grid(row=2, column=0, sticky='w', **pad)
    self.task_var = tk.StringVar(value='classification')
    ttk.Combobox(pfrm, textvariable=self.task_var, values=['classification', 'segmentation'], state='readonly', width=16).grid(row=2, column=1, **pad)

    ttk.Label(pfrm, text='Stride (seg only)').grid(row=2, column=2, sticky='w', **pad)
    self.stride_var = tk.IntVar(value=0)
    ttk.Spinbox(pfrm, from_=0, to=4096, increment=16, textvariable=self.stride_var, width=10).grid(row=2, column=3, **pad)

    # Progress
    self.prog = ttk.Progressbar(frm, orient='horizontal', mode='determinate')
    self.prog.grid(row=6, column=0, columnspan=3, sticky='we', **pad)
    self.prog_label = ttk.Label(frm, text='Idle')
    self.prog_label.grid(row=7, column=0, columnspan=3, sticky='w', **pad)

    # Controls
    cfrm = ttk.Frame(frm)
    cfrm.grid(row=8, column=0, columnspan=3, sticky='we', **pad)
    ttk.Button(cfrm, text='Start', command=self._start).pack(side='left', padx=4)
    ttk.Button(cfrm, text='Cancel', command=self._cancel).pack(side='left', padx=4)

    # Log
    lfrm = ttk.LabelFrame(frm, text='Log')
    lfrm.grid(row=9, column=0, columnspan=3, sticky='nsew', **pad)
    frm.rowconfigure(9, weight=1)
    frm.columnconfigure(1, weight=1)
    self.log = tk.Text(lfrm, height=10)
    self.log.pack(fill='both', expand=True)

    self.after(100, self._poll_log)

    def _browse_input(self):
        path = filedialog.askopenfilename(title='Select slide or choose folder in next dialog', filetypes=[('Slides', '*.svs *.dcm'), ('All', '*.*')])
        if path:
            self.input_var.set(path)
        else:
            folder = filedialog.askdirectory(title='Select folder with slides')
            if folder:
                self.input_var.set(folder)

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(title='Select checkpoint', filetypes=[('PyTorch Checkpoint', '*.pth *.pt'), ('All', '*.*')])
        if path:
            self.ckpt_var.set(path)

    def _browse_ckpt_cls(self):
        path = filedialog.askopenfilename(title='Select ROI classifier checkpoint', filetypes=[('PyTorch Checkpoint', '*.pth *.pt'), ('All', '*.*')])
        if path:
            self.ckpt_cls_var.set(path)

    def _browse_ann(self):
        folder = filedialog.askdirectory(title='Select annotations folder')
        if folder:
            self.ann_var.set(folder)

    def _browse_out(self):
        path = filedialog.asksaveasfilename(title='Save results JSON', defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            self.out_var.set(path)

    def _log(self, msg: str):
        self._log_queue.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log.insert('end', msg + '\n')
                self.log.see('end')
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _on_progress(self, done: int, total: int):
        total = max(1, int(total))
        done = min(done, total)
        self.prog['maximum'] = total
        self.prog['value'] = done
        self.prog_label.config(text=f'Processing patches: {done}/{total}')
        self.update_idletasks()

    def _start(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo('Running', 'Inference already in progress')
            return
        self._cancel_event.clear()
        self.prog['value'] = 0
        self.prog_label.config(text='Starting…')
        self.log.delete('1.0', 'end')

        def runner():
            try:
                summary = infer_mod.infer(
                    input_path=self.input_var.get(),
                    checkpoint=self.ckpt_var.get(),
                    output_json=self.out_var.get(),
                    batch_size=int(self.bs_var.get()),
                    patch_size=int(self.ps_var.get()),
                    patches_per_slide=int(self.pps_var.get()),
                    svs_level=int(self.level_var.get()),
                    num_workers=int(self.nw_var.get()),
                    annotations_dir=(self.ann_var.get() or None),
                    device=self.dev_var.get(),
                    progress_cb=self._on_progress,
                    cancel_event=self._cancel_event,
                    task=self.task_var.get(),
                    stride=(int(self.stride_var.get()) or None),
                    cls_checkpoint=(self.ckpt_cls_var.get() or None),
                )
                self._log(f"Done. Slides: {summary['num_slides']} | Output: {self.out_var.get()}")
                self.prog_label.config(text='Done')
            except Exception as e:
                self._log(f'Error: {e}')
                self.prog_label.config(text='Error')

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def _cancel(self):
        if self._thread and self._thread.is_alive():
            self._cancel_event.set()
            self._log('Cancellation requested… will stop after current batch')
        else:
            self._log('Nothing to cancel')


def main():
    app = InferenceGUI()
    app.mainloop()


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
