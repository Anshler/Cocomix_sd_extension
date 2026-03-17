function cocomixCopyCaption(text) {
    if (!text) return;
    const btn = document.activeElement;
    const flash = () => {
        if (!btn._origLabel) btn._origLabel = btn.textContent;
        clearTimeout(btn._resetTimer);
        btn.textContent = 'Copied!';
        btn.style.background = '#22c55e';
        btn.style.color = '#fff';
        btn.style.transition = 'background 0.4s, color 0.4s';
        btn._resetTimer = setTimeout(() => {
            btn.textContent = btn._origLabel;
            btn._origLabel = null;
            btn.style.background = '';
            btn.style.color = '';
        }, 1500);
    };
    const fallback = (t) => {
        const ta = document.createElement('textarea');
        ta.value = t;
        Object.assign(ta.style, { position: 'fixed', opacity: '0' });
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        if (document.execCommand('copy')) flash();
        document.body.removeChild(ta);
    };
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(flash).catch(() => fallback(text));
    } else {
        fallback(text);
    }
}
