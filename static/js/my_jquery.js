

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

    var optarray = $("#forecast_Time").children('option').map(function() {
        return {
            "value": this.value,
            "option": "<option value='" + this.value + "'>" + this.text + "</option>"
        }
    })

    $("#model_Selection").change(function() {
        $("#forecast_Time").children('option').remove();
        var addoptarr = [];
        for (i = 0; i < optarray.length; i++) {
            if (optarray[i].value.indexOf($(this).val()) > -1) {
                addoptarr.push(optarray[i].option);
            }
        }
        $("#forecast_Time").html(addoptarr.join(''))
    }).change();



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



