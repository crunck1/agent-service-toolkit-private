// Funzione per evidenziare gli elementi con una bounding box e includere il testo, ariaLabel e tipo
function markPage() {
    // Trova tutti gli elementi della pagina che vuoi evidenziare
    const selectors = [
        'div', 'span', 'a', 'button', 'input', 'textarea', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'ul', 'ol', 'li', 'img', 'table', 'tr', 'td', 'th', 'form', 'iframe', 'section', 'article'
    ];
    const elements = document.querySelectorAll(selectors.join(', '));

    // Array per memorizzare le bounding box e il testo associato
    var bboxes = [];

    // Itera attraverso gli elementi e aggiungi una bounding box
    elements.forEach(function(element) {
        // Ottieni le coordinate dell'elemento
        var rect = element.getBoundingClientRect();

        // Crea un div per la bounding box
        var box = document.createElement('div');
        box.style.position = 'absolute';
        box.style.border = '2px solid red';
        box.style.top = rect.top + 'px';
        box.style.left = rect.left + 'px';
        box.style.width = rect.width + 'px';
        box.style.height = rect.height + 'px';
        box.style.pointerEvents = 'none'; // Impedisce l'interazione con la bounding box
        box.classList.add('bounding-box');

        // Aggiungi il div al body
        document.body.appendChild(box);

        // Ottieni il testo dell'elemento (ignora elementi senza testo significativo)
        var text = element.innerText || element.textContent || '';

        // Ottieni l'aria-label dell'elemento se disponibile
        var ariaLabel = element.getAttribute('aria-label') || '';

        // Ottieni il tipo di elemento (ad es. 'button', 'div', ecc.)
        var el_type = element.tagName.toLowerCase();

        // Memorizza le coordinate, il testo, l'ariaLabel e il tipo
        bboxes.push({
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height,
            text: text.trim(),         // Aggiungi il testo dell'elemento
            ariaLabel: ariaLabel.trim(), // Aggiungi aria-label se presente
            type: el_type,              // Aggiungi il tipo di elemento
        });
    });

    // Restituisci le bounding box con il testo associato
    return bboxes;
}

// Funzione per rimuovere le bounding box
function unmarkPage() {
    var boxes = document.querySelectorAll('.bounding-box');
    boxes.forEach(function(box) {
        box.remove();
    });
}
