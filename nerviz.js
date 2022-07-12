requirejs['jquery'];


$('.token').on('mouseenter', function(e) {
    $(this).css('background-color', 'cyan');
});

$('.token').on('mouseleave', function(e) {
    $(this).css('background-color', 'transparent');
});


// @param attention an array of attention for target tokens
function refreshAttention(attention) {
    $('.target').each(function (i, element){
	blue = attention[i] * 255;
	blue_hex = blue.toString(16);
	if (blue_hex.length == 1) {
	    blue_hex = "0" + blue_hex;
	}
	element.css("background-color", "#0000" + blue_hex);
    });
}
