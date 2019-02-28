

$(document).ready(function(e){

    $("#submitButton2").on('click',function(){
		$(this).html('Downloading...');
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });


    $("#buttonTryAgain").on('click',function(){
		parent.history.back();
        return false;
    });

    $('[data-toggle="popover"]').popover();
});

$(function () {
	$('#datetimepicker4').datetimepicker({
	format: 'L'
  });
});

function modal(){
   $('.modal').modal('show');
   setTimeout(function () {
    console.log('hejsan');
    $('.modal').modal('hide');
   }, 2000);

}



