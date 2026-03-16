/**
 * Preview panel: shows output images/videos and latent previews.
 */

class SFPreview {
    api: SFApiInstance;
    panel: HTMLElement;
    content: HTMLElement;
    closeBtn: HTMLElement;
    _currentBlobUrl: string | null;

    constructor(api: SFApiInstance) {
        this.api = api;
        this.panel = document.getElementById('preview-panel')!;
        this.content = document.getElementById('preview-content')!;
        this.closeBtn = document.getElementById('preview-close')!;
        this._currentBlobUrl = null;

        this.closeBtn.addEventListener('click', () => this.hide());

        // Make header draggable
        this._initDrag();

        // Listen for binary preview (latent)
        this.api.on('preview', (data: WSEventData) => {
            if (data && data.url) {
                this.showImage(data.url, true);
            }
        });
    }

    _initDrag(): void {
        const header = document.getElementById('preview-header');
        let dragging = false, startX: number = 0, startY: number = 0, startLeft: number = 0, startTop: number = 0;

        header!.addEventListener('mousedown', (e) => {
            if ((e.target as HTMLElement).tagName === 'BUTTON') return;
            dragging = true;
            const rect = this.panel.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            // Switch from bottom/right positioning to top/left
            this.panel.style.left = rect.left + 'px';
            this.panel.style.top = rect.top + 'px';
            this.panel.style.bottom = 'auto';
            this.panel.style.right = 'auto';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            this.panel.style.left = (startLeft + dx) + 'px';
            this.panel.style.top = (startTop + dy) + 'px';
        });

        document.addEventListener('mouseup', () => {
            dragging = false;
        });
    }

    showImage(url: string, isBlob?: boolean): void {
        this.content.innerHTML = '';
        this._revokeBlobUrl();

        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Preview';
        if (isBlob) this._currentBlobUrl = url;

        this.content.appendChild(img);
        this.panel.classList.remove('hidden');
    }

    showVideo(url: string): void {
        this.content.innerHTML = '';
        this._revokeBlobUrl();

        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;

        this.content.appendChild(video);
        this.panel.classList.remove('hidden');
    }

    hide(): void {
        this.panel.classList.add('hidden');
        this._revokeBlobUrl();
    }

    _revokeBlobUrl(): void {
        if (this._currentBlobUrl) {
            URL.revokeObjectURL(this._currentBlobUrl);
            this._currentBlobUrl = null;
        }
    }
}
