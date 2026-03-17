(function () {
    const style = document.createElement('style');
    style.textContent = `
        .mdl-label-row { display: flex; align-items: center; gap: 6px; font-size: 15px; font-weight: 500; padding: 4px 0; }
        .mdl-tip { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; background: #6b7280; color: #fff; font-size: 10px; font-weight: bold; cursor: help; position: relative; flex-shrink: 0; user-select: none; }
        .mdl-tip-box { display: none; width: 300px; background: #1f2937; color: #f9fafb; padding: 10px 12px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 12px; line-height: 1.6; font-style: normal; font-weight: normal; position: absolute; z-index: 9999; left: 0; top: 130%; }
        .mdl-tip:hover .mdl-tip-box { display: block; }
        #cocomix-word-count p { font-size: 11px; color: #6b7280; margin: 2px 0 0; }
    `;
    document.head.appendChild(style);
})();
