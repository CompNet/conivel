requirejs['jquery'];

window.currentSource = 0;
window.attentions = null;

// note : use the global window.attentions variable, which is a 2d
// array of shape (source, target)
function refreshAttentions() {
    
    if (window.attentions === null) {
	return;
    }

    $('.target').each(function (i, element) {
	// [0-1]
	let attention = window.attentions[window.currentSource][i];
	// [155-255]
	let red = Math.floor(255 - attention * 100);
	let red_hex = red.toString(16);
	if (red_hex.length === 1) {
	    red_hex = "0" + red_hex;
	}
	$(element).css("background-color", "#" + red_hex + "ccff");
    });
    
}
window.refreshAttentions = refreshAttentions;


$('.source').on('mouseenter', function(e) {
    $(this).css('background-color', 'cyan');
    window.currentSource = $('.source').index(this);
    window.refreshAttentions();
});

$('.source').on('mouseleave', function(e) {
    $(this).css('background-color', 'transparent');
});

