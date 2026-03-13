/**
 * Preview panel: shows output images/videos and latent previews.
 */
class SFPreview {
    constructor(api) {
        this.api = api;
        this.panel = document.getElementById('preview-panel');
        this.content = document.getElementById('preview-content');
        this.closeBtn = document.getElementById('preview-close');
        this._currentBlobUrl = null;

        this.closeBtn.addEventListener('click', () => this.hide());

        // Listen for binary preview (latent)
        this.api.on('preview', (data) => {
            if (data && data.url) {
                this.showImage(data.url, true);
            }
        });
    }

    showImage(url, isBlob) {
        this.content.innerHTML = '';
        this._revokeBlobUrl();

        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Preview';
        if (isBlob) this._currentBlobUrl = url;

        this.content.appendChild(img);
        this.panel.classList.remove('hidden');
    }

    showVideo(url) {
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

    hide() {
        this.panel.classList.add('hidden');
        this._revokeBlobUrl();
    }

    _revokeBlobUrl() {
        if (this._currentBlobUrl) {
            URL.revokeObjectURL(this._currentBlobUrl);
            this._currentBlobUrl = null;
        }
    }
}
