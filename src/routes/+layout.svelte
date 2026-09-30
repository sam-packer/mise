<script lang="ts">
	// Apply shared styles to every route.
	import './layout.css';

	let { children } = $props();
</script>

{@render children()}
