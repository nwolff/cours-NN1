<script type="ts">
	import numbersImage from '$lib/assets/neural-numbers.png';
	import { base } from '$app/paths';
</script>

<h1 class="px-8 text-3xl">Aider un réseau de neurones à apprendre</h1>
<img alt="Chiffres" src={numbersImage} class="pt-8 size-10/12" />
<div class="p-8 flex flex-row">
	<div class="basis-1/4">
		<a class="link link-primary" href="https://forms.office.com/e/Nt1PhkpX76" target="_blank"
			>Questionnaire avant</a
		>
	</div>
	<div class="basis-1/4">
		<a class="link link-error" href="https://forms.office.com/e/wb66rDKYHj" target="_blank"
			>Questionnaire après</a
		>
	</div>
	<div class="basis-1/4">
		<a class="link" href="{base}/about">A propos</a>
	</div>
</div>
